import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from timm.layers import trunc_normal_
from typing import Tuple, Optional


def get_relative_distances(window_size):
    """
    Compute relative distances between positions in a 1D window.

    Args:
        window_size (int): The size of the window.

    Returns:
        torch.Tensor: A tensor containing the relative position indices.
    """
    indices = torch.arange(window_size)
    relative_indices = indices[None, :] - indices[:, None]
    return relative_indices + window_size - 1


def get_1d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create 1D sinusoidal positional embeddings using PyTorch, with a calculation
    method similar to the provided NumPy snippet.

    Args:
        embed_dim: Dimensionality of the embeddings. Must be even.
        grid_size: Number of positions for which embeddings are generated.

    Returns:
        A PyTorch nn.Parameter containing the positional embeddings.
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Generate position indices
    pos = torch.arange(grid_size).float().unsqueeze(1)

    # Calculate omega based on embed_dim (using the provided mathematical approach)
    omega = torch.arange(embed_dim // 2).float()
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))

    # Calculate the outer product, similar to np.einsum('m,d->md', pos, omega)
    out = pos * omega

    # Generate sinusoidal embeddings
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    # Concatenate sin and cos embeddings
    emb = torch.cat([emb_sin, emb_cos], dim=1)

    # Wrap the embeddings in nn.Parameter, with requires_grad=False
    pos_embed = torch.nn.Parameter(emb.unsqueeze(0), requires_grad=False)
    return pos_embed


class MLP(nn.Module):
    """
    Pre-norm MLP head:
      LayerNorm -> [Linear -> GELU -> (Dropout)] x L -> Linear (to out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Optional[Tuple[int, ...]] = (512,),  # e.g., (512,) or (512, 256)
        dropout: Optional[float] = 0.0,
        use_ln: Optional[bool] = True,
    ):
        super().__init__()

        layers = []

        if use_ln:
            layers.append(nn.LayerNorm(in_dim))

        dim = in_dim
        if hidden is not None:
            for h in hidden:
                layers.append(nn.Linear(dim, h))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                dim = h

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


class GRN(nn.Module):
    """
    Gated Residual Normalization (GRN) module.

    This module normalizes input features and applies learnable scaling and shifting.

    Args:
        dim (int): Dimensionality of input features.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        """
        Forward pass of the GRN module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim, sequence_length).

        Returns:
            torch.Tensor: Transformed tensor with normalized scaling and bias applied.
        """
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class CoPE(nn.Module):
    """
    Contextual Positional Encoding (CoPE) module.

    This module computes position embeddings dynamically based on attention logits,
    allowing for flexible and adaptive positional encoding.

    Args:
        npos_max (int): Maximum number of discrete positions.
        head_dim (int): Dimensionality of the attention head.
    """

    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        """
        Forward pass for the CoPE module.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_length, head_dim).
            attn_logits (torch.Tensor): Attention logits of shape (batch_size, num_heads, seq_length, seq_length).

        Returns:
            torch.Tensor: Contextual positional encodings of shape (batch_size, num_heads, seq_length, head_dim).
        """
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)

        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor

        return logits_ceil * w + logits_floor * (1 - w)


class ECGWindowAttention(nn.Module):
    """
    Window-based Self-Attention module for ECG signals.

    This module performs self-attention within fixed-size windows and optionally applies
    relative positional encoding (RPE), cyclic shifts, and cosine similarity-based attention.

    Args:
        dim (int): Feature dimension.
        heads (int): Number of attention heads.
        window_size (int): Size of each attention window.
        rpe_type (str): Type of RPE to use. Options are None, "cope", "default", or "combined".
        shift (bool): Whether to apply cyclic shift for shifted window attention.
        shift_size (int): Cyclic shift size. Defaults to 'window_size//2'.
        cosine_sim (bool): Whether to use cosine similarity for attention computation.
    """

    def __init__(
        self,
        dim,
        heads,
        window_size,
        rpe_type="cope",
        shift=False,
        shift_size=None,
        cosine_sim=True,
    ):
        super().__init__()
        head_dim = dim // heads
        self.heads = heads
        self.scale = head_dim**0.5
        self.window_size = window_size
        self.shift = shift
        self.shift_size = shift_size
        self.rpe_type = rpe_type
        self.cosine_sim = cosine_sim

        self.use_rpe = rpe_type is not None

        # Linear projection for queries, keys, and values
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        if self.cosine_sim:
            # Learnable temperature parameter for cosine similarity attention
            self.tau = nn.Parameter(torch.ones(1), requires_grad=True)

        # Relative positional encoding setup
        if self.use_rpe:
            if self.rpe_type not in ["default", "cope", "combined"]:
                raise ValueError("Invalid value for 'rpe_type'.")

            if self.rpe_type in ["cope", "combined"]:
                self.cope = CoPE(window_size, head_dim)

            if self.rpe_type in ["default", "combined"]:
                self.register_buffer(
                    "relative_indices",
                    get_relative_distances(window_size),
                    persistent=False,
                )
                self.pos_embedding = nn.Parameter(torch.zeros(2 * window_size - 1))

            if self.rpe_type == "combined":
                self.rpe_alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.to_out = nn.Linear(dim, dim)

    def _create_mask_1d(self, window_size, displacement):
        mask = torch.zeros(window_size, window_size)
        if displacement > 0:
            mask[:-displacement, -displacement:] = float("-inf")
            mask[-displacement:, :-displacement] = float("-inf")

        return mask

    def _get_pos_embedding(self, query, attn_logits):
        """
        Compute relative positional encodings based on the selected RPE type.

        Args:
            query (torch.Tensor): Query tensor (batch_size, num_heads, num_windows, seq_length, head_dim).
            attn_logits (torch.Tensor): Attention logits (batch_size, num_heads, num_windows, seq_length, seq_length).

        Returns:
            torch.Tensor: Positional encoding tensor.
        """
        if self.rpe_type == "cope":
            return self.cope(query, attn_logits)

        else:
            if self.rpe_type == "combined":
                cope_embeddings = self.cope(query, attn_logits)
                rpe_embeddings = self.pos_embedding[self.relative_indices]

                alpha = torch.sigmoid(self.rpe_alpha)
                embeddings = alpha * rpe_embeddings + (1 - alpha) * cope_embeddings

                return embeddings

            else:
                return self.pos_embedding[self.relative_indices]

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_length, feature_dim).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        b, l, _ = x.shape

        if l % self.window_size != 0:
            raise ValueError(
                f"Sequence length {l} must be divisible by window_size {self.window_size}."
            )

        shift_size = (
            self.shift_size or self.window_size // 2
        )  # Shift size defaults to window_size//2 if not specified
        nw_l = l // self.window_size  # Number of windows along the sequence length

        if self.shift:
            x_shift = torch.roll(x, shifts=-shift_size, dims=1)
        else:
            x_shift = x

        qkv = self.to_qkv(x_shift)
        q, k, v = (
            qkv.view(b, nw_l, self.window_size, self.heads, -1)
            .permute(0, 3, 1, 2, 4)
            .chunk(3, dim=4)
        )

        if self.cosine_sim:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            attn = einsum("b h w i d, b h w j d -> b h w i j", q, k) / self.tau
        else:
            attn = einsum("b h w i d, b h w j d -> b h w i j", q, k) / self.scale

        # Apply attention mask if cyclic shift is used
        if self.shift:
            attn_mask = self._create_mask_1d(self.window_size, shift_size).to(
                attn.device
            )
            attn[:, :, -1] += attn_mask

        if self.use_rpe:
            attn += self._get_pos_embedding(query=q, attn_logits=attn)

        attn = attn.softmax(dim=-1)

        out = einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        out = rearrange(out, "b h w l d -> b (w l) (h d)")
        out = self.to_out(out)

        if self.shift:
            out = torch.roll(out, shifts=shift_size, dims=1)

        return out


class ECGSwinBlock(nn.Module):
    """
    Swin Transformer Block for ECG signal processing.

    This block applies window-based self-attention followed by a feed-forward network.
    It supports optional shifted windows and relative positional encoding (RPE).

    Args:
        dim (int): Feature dimension.
        heads (int): Number of attention heads.
        expansion_factor (int): Expansion factor for the hidden channel dimension.
        p_drop (float): Dropout probability used in the feed forward net.
        shift (bool): Whether to apply cyclic shift for shifted window attention.
        shift_size (int): Cyclic shift size. Defaults to 'window_size//2'.
        window_size (int): Size of the attention window.
        rpe_type (str): Type of RPE to use ("cope", "default", "combined").
        cosine_sim (bool): Whether to use cosine similarity for attention computation.
    """

    def __init__(
        self,
        dim,
        heads,
        expansion_factor,
        p_drop,
        shift,
        shift_size,
        window_size,
        rpe_type,
        cosine_sim,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = ECGWindowAttention(
            dim=dim,
            heads=heads,
            window_size=window_size,
            rpe_type=rpe_type,
            shift=shift,
            shift_size=shift_size,
            cosine_sim=cosine_sim,
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ff = MLP(
            in_dim=dim,
            hidden=(dim * expansion_factor,),
            out_dim=dim,
            dropout=p_drop,
            use_ln=False,
        )

        self.dim = dim
        self.window_size = window_size

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        # Apply first normalization and self-attention
        residual = x
        out = self.ln1(x)
        out = self.attn(out)
        out = out + residual  # Add residual connection

        # Apply second normalization and feed-forward network
        residual = out
        out = self.ln2(out)
        out = self.ff(out)
        out = out + residual  # Add residual connection

        return out


class ECGConvMerge(nn.Module):
    """
    Convolutional patch merging module for ECG signals.

    This module receives a sequence of ECG patch embeddings, downsamples
    the sequence with a convolutional branch, and combines it with a
    downsampled residual path. A second  feed-forward-style convolutional
    block is then applied with another residual connection.

    Args:
        in_c (int): Number of input channels/features per patch.
        out_c (int): Number of output channels/features after patch merging.
        seq_len (int): Output sequence length after downsampling.
        downsample_factor (int): Factor used to reduce the temporal length.
        expansion_factor (int): Expansion factor for the hidden channel dimension.
        p_drop (float): Dropout probability used in the first convolutional block.
        conv_config (dict): Keyword arguments passed to the initial Conv1d layer.
    """

    def __init__(
        self,
        in_c,
        out_c,
        seq_len,
        downsample_factor,
        expansion_factor,
        p_drop,
        conv_config,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.downsample_factor = downsample_factor

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Conv1d(in_c, out_c, **conv_config),  # Downsample
            nn.LayerNorm([out_c, seq_len]),  # LayerNorm
            nn.Conv1d(
                out_c, out_c * expansion_factor, kernel_size=1, stride=1
            ),  # Pointwise conv
            nn.GELU(),  # Non-linearity
            nn.Dropout(p_drop),  # Dropout
            GRN(out_c * expansion_factor),  # GRN for feature scaling
            nn.Conv1d(
                out_c * expansion_factor, out_c, kernel_size=1, stride=1
            ),  # Pointwise conv
        )

        # Skip connection for residual learning
        self.skip_connection = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=downsample_factor, stride=downsample_factor
            ),  # Downsample
            nn.Conv1d(in_c, out_c, kernel_size=1, stride=1),  # Pointwise conv
        )

        # Final convolutional branch
        self.final_layer = nn.Sequential(
            nn.LayerNorm([out_c, seq_len]),  # LayerNorm
            nn.Conv1d(
                out_c, out_c * expansion_factor, kernel_size=1, stride=1
            ),  # Pointwise conv
            nn.GELU(),  # Non-linearity
            GRN(out_c * expansion_factor),  # GRN
            nn.Conv1d(
                out_c * expansion_factor, out_c, kernel_size=1, stride=1
            ),  # Pointwise conv
        )

    def forward(self, x):
        """
        Forward pass of ConvMerge.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, out_channels).
        """
        # Convert from (B, L, C) to (B, C, L)
        out = x.permute(0, 2, 1).contiguous()

        # Apply skip connection
        residual = self.skip_connection(out)
        out = self.input_layer(out)
        out = residual + out  # Residual connection

        # Apply second convolutional block
        residual = out
        out = self.final_layer(out)
        out = residual + out  # Residual connection

        # Convert to (B, L, C) always
        out = out.permute(0, 2, 1).contiguous()

        return out


class ECGStageModule(nn.Module):
    """
    ECG stage composed of convolutional patch merging followed by pairs of
    regular and shifted Swin Transformer blocks.

    The module optionally applies absolute positional embeddings.

    Args:
        in_c (int): Number of input channels/features.
        out_c (int): Number of output channels/features.
        downsample_factor (int): Temporal downsampling factor used during patch merging.
        expansion_factor (int): Hidden dimension expansion factor used in MLP-style blocks.
        merge_p_drop (float): Dropout probability used in the patch merging module.
        merge_conv_config (dict): Arguments for the initial Conv1d merge layer.
        layers (int): Number of Swin blocks. Must be even.
        num_heads (int): Number of attention heads.
        window_size (int): Attention window size.
        attn_p_drop (float): Dropout probability used in Swin feed-forward blocks.
        rpe_type (str): Relative positional encoding type.
        cosine_sim (bool): Whether to use cosine similarity attention.
        shift (bool): Whether to use shifted-window attention.
        shift_size (int | None): Shift size for shifted attention.
        use_ape (bool): Whether to use absolute positional embeddings.
        merged_seq_len (int): Sequence length after patch merging.
    """

    def __init__(
        self,
        in_c,
        out_c,
        downsample_factor,
        expansion_factor,
        merge_p_drop,
        merge_conv_config,
        layers,
        num_heads,
        window_size,
        attn_p_drop,
        rpe_type,
        cosine_sim,
        shift,
        shift_size,
        use_ape,
        merged_seq_len,
    ):
        super().__init__()
        assert (
            layers % 2 == 0
        ), "Stage layers need to be divisible by 2 for regular and shifted blocks."

        self.use_ape = use_ape

        # Patch merging for ECG signals
        self.patch_merge = ECGConvMerge(
            in_c=in_c,
            out_c=out_c,
            seq_len=merged_seq_len,
            downsample_factor=downsample_factor,
            expansion_factor=expansion_factor,
            p_drop=merge_p_drop,
            conv_config=merge_conv_config,
        )

        # Absolute positional embeddings (APE) for later stages
        if self.use_ape:
            self.abs_pos_emb = get_1d_sincos_pos_embed(out_c, merged_seq_len)

        # Swin Transformer Blocks for ECG signals
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(
                nn.ModuleList(
                    [
                        ECGSwinBlock(
                            dim=out_c,
                            heads=num_heads,
                            p_drop=attn_p_drop,
                            expansion_factor=expansion_factor,
                            cosine_sim=cosine_sim,
                            shift=False,  # Regular window-based attention
                            shift_size=shift_size,
                            window_size=window_size,
                            rpe_type=rpe_type,
                        ),
                        ECGSwinBlock(
                            dim=out_c,
                            heads=num_heads,
                            p_drop=attn_p_drop,
                            expansion_factor=expansion_factor,
                            cosine_sim=cosine_sim,
                            shift=shift,  # Shifted window-based attention
                            shift_size=shift_size,
                            window_size=window_size,
                            rpe_type=rpe_type,
                        ),
                    ]
                )
            )

    def forward(self, x):
        """
        Forward pass for the ECG stage module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim).

        Returns:
            torch.Tensor: Processed tensor after patch merging and Swin Transformer blocks.
        """
        # Patch Merge
        out = self.patch_merge(x)

        # Add absolute positional embeddings if enabled and in later stages
        if self.use_ape:
            out += self.abs_pos_emb

        # Apply all Swin Transformer blocks in sequence
        for block1, block2 in self.layers:
            out = block1(out)
            out = block2(out)

        return out


class ECGHiTNeXt(nn.Module):
    """
    Hierarchical ECG model built from multiple ECGStageModule blocks.

    Each stage applies convolutional patch merging followed by local Swin-style
    attention blocks. Stage parameters can be either scalars, reused across all
    stages, or lists/tuples with length equal to `num_stages`.

    Args:
        ecg_size (tuple): Input shape as (seq_len, channels).
        num_stages (int): Number of hierarchical stages.

        hidden_dim (int or list): Output feature dimension per stage.
        layers (int or list): Number of Swin blocks per stage. Must be even.
        heads (int or list): Number of attention heads per stage.

        downsample_factor (int or list): Temporal downsampling factor per stage.
        window_size (int or list): Attention window size per stage.
        expansion_factor (int or list): Hidden expansion factor in merge/MLP blocks.

        merge_conv_config (dict or list): Conv1d config for patch merging.
        merge_p_drop (float or list): Dropout used in patch merging.
        attn_p_drop (float or list): Dropout used in Swin feed-forward blocks.

        rpe_type (str or list): Relative positional encoding type per stage.
        use_ape (bool or list): Whether to use absolute positional embeddings.
        cosine_sim (bool or list): Whether to use cosine similarity attention.
        shift (bool or list): Whether to use shifted-window attention.
        shift_size (int, None, or list): Shift size for shifted attention.

        apply_out_mlp (bool): Whether to apply the output MLP head.
        out_mlp_hidden_dim (tuple or None): Hidden dimensions for the output MLP.
        out_mlp_out_dim (int): Output dimension of the MLP head.
        out_mlp_dropout (float): Dropout used in the output MLP.
        out_mlp_norm (bool): Whether to use LayerNorm in the output MLP.
    """

    def __init__(
        self,
        ecg_size,
        num_stages,
        hidden_dim,
        layers,
        heads,
        downsample_factor,
        window_size,
        expansion_factor,
        merge_conv_config,
        merge_p_drop,
        attn_p_drop,
        rpe_type,
        use_ape,
        cosine_sim,
        shift,
        shift_size,
        apply_out_mlp,
        out_mlp_hidden_dim,
        out_mlp_out_dim,
        out_mlp_dropout,
        out_mlp_norm,
    ):
        super().__init__()

        self.ecg_size = ecg_size
        self.num_stages = num_stages
        self.apply_out_mlp = apply_out_mlp

        self.hidden_dim = self._to_stage_list(hidden_dim, num_stages, "hidden_dim")
        self.layers = self._to_stage_list(layers, num_stages, "layers")
        self.heads = self._to_stage_list(heads, num_stages, "heads")

        self.downsample_factor = self._to_stage_list(
            downsample_factor, num_stages, "downsample_factor"
        )
        self.window_size = self._to_stage_list(window_size, num_stages, "window_size")
        self.expansion_factor = self._to_stage_list(
            expansion_factor, num_stages, "expansion_factor"
        )

        self.merge_conv_config = self._to_stage_list(
            merge_conv_config, num_stages, "merge_conv_config"
        )
        self.merge_p_drop = self._to_stage_list(
            merge_p_drop, num_stages, "merge_p_drop"
        )
        self.attn_p_drop = self._to_stage_list(attn_p_drop, num_stages, "attn_p_drop")

        self.rpe_type = self._to_stage_list(rpe_type, num_stages, "rpe_type")
        self.use_ape = self._to_stage_list(use_ape, num_stages, "use_ape")
        self.cosine_sim = self._to_stage_list(cosine_sim, num_stages, "cosine_sim")
        self.shift = self._to_stage_list(shift, num_stages, "shift")
        self.shift_size = self._to_stage_list(shift_size, num_stages, "shift_size")

        self.stages = nn.ModuleList()
        self.merged_seq_lens = []

        seq_len, in_c = ecg_size

        for stage_idx in range(num_stages):
            out_c = self.hidden_dim[stage_idx]
            num_heads = self.heads[stage_idx]
            current_downsample_factor = self.downsample_factor[stage_idx]
            current_window_size = self.window_size[stage_idx]

            assert self.layers[stage_idx] % 2 == 0, f"layers[{stage_idx}] must be even."

            assert (
                out_c % num_heads == 0
            ), f"hidden_dim[{stage_idx}] must be divisible by heads[{stage_idx}]."

            assert (
                seq_len % current_downsample_factor == 0
            ), f"seq_len before stage {stage_idx} must be divisible by downsample_factor."

            seq_len = seq_len // current_downsample_factor

            assert (
                seq_len % current_window_size == 0
            ), f"merged_seq_len at stage {stage_idx} must be divisible by window_size."

            self.merged_seq_lens.append(seq_len)

            self.stages.append(
                ECGStageModule(
                    in_c=in_c,
                    out_c=out_c,
                    downsample_factor=current_downsample_factor,
                    expansion_factor=self.expansion_factor[stage_idx],
                    merge_p_drop=self.merge_p_drop[stage_idx],
                    merge_conv_config=self.merge_conv_config[stage_idx],
                    layers=self.layers[stage_idx],
                    num_heads=num_heads,
                    window_size=current_window_size,
                    attn_p_drop=self.attn_p_drop[stage_idx],
                    rpe_type=self.rpe_type[stage_idx],
                    cosine_sim=self.cosine_sim[stage_idx],
                    shift=self.shift[stage_idx],
                    shift_size=self.shift_size[stage_idx],
                    use_ape=self.use_ape[stage_idx],
                    merged_seq_len=seq_len,
                )
            )

            in_c = out_c

        if self.apply_out_mlp:
            self.out_mlp = MLP(
                in_dim=self.hidden_dim[-1],
                out_dim=out_mlp_out_dim,
                hidden=out_mlp_hidden_dim,
                dropout=out_mlp_dropout,
                use_ln=out_mlp_norm,
            )
        else:
            self.out_mlp = None

        self.apply(self._init_weights)

    @staticmethod
    def _to_stage_list(value, num_stages, name):
        if isinstance(value, (list, tuple)):
            if len(value) != num_stages:
                raise ValueError(
                    f"'{name}' must have length {num_stages}, but got {len(value)}."
                )
            return list(value)

        if isinstance(value, dict):
            return [value.copy() for _ in range(num_stages)]

        return [value for _ in range(num_stages)]

    def forward(self, x):
        """
        Forward pass through all stages.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C).

        Returns:
            torch.Tensor: Final sequence features, or output head predictions if enabled.
        """
        out = x
        for stage_idx in range(self.num_stages):
            out = self.stages[stage_idx](out)

        if self.apply_out_mlp:
            out = out.mean(dim=1)
            out = self.out_mlp(out)

        return out

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, CoPE):
            trunc_normal_(m.pos_emb, std=0.02)

        elif isinstance(m, ECGWindowAttention):
            if hasattr(m, "pos_embedding"):
                trunc_normal_(m.pos_embedding, std=0.02)

            if hasattr(m, "rpe_alpha"):
                nn.init.constant_(m.rpe_alpha, 0)

            if hasattr(m, "tau"):
                nn.init.constant_(m.tau, 1.0)
