# tests/test_ecg_hitnext.py

import pytest
import torch
import torch.nn.functional as F

from ecg_hit_next import (
    ECGHiTNeXt,
    ECGWindowAttention,
    ECGConvMerge,
    ECGStageModule,
    GRN,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU is required for these tests.",
)


def _make_model(
    *,
    apply_out_mlp=True,
    rpe_type="default",
    use_ape=True,
    cosine_sim=False,
    shift=True,
):
    return ECGHiTNeXt(
        ecg_size=(256, 12),
        num_stages=2,
        hidden_dim=[32, 64],
        layers=[2, 2],
        heads=[4, 4],
        downsample_factor=[4, 4],
        window_size=[8, 4],
        expansion_factor=2,
        merge_conv_config=[
            {"kernel_size": 4, "stride": 4, "padding": 0},
            {"kernel_size": 4, "stride": 4, "padding": 0},
        ],
        merge_p_drop=0.1,
        attn_p_drop=0.1,
        rpe_type=rpe_type,
        use_ape=use_ape,
        cosine_sim=cosine_sim,
        shift=shift,
        shift_size=None,
        apply_out_mlp=apply_out_mlp,
        out_mlp_hidden_dim=(32,),
        out_mlp_out_dim=5,
        out_mlp_dropout=0.1,
        out_mlp_norm=True,
    ).cuda()


@pytest.mark.parametrize("rpe_type", [None, "default", "cope", "combined"])
@pytest.mark.parametrize("cosine_sim", [False, True])
@pytest.mark.parametrize("shift", [False, True])
def test_attention_forward_multiple_configs(rpe_type, cosine_sim, shift):
    attn = ECGWindowAttention(
        dim=32,
        heads=4,
        window_size=8,
        rpe_type=rpe_type,
        shift=shift,
        shift_size=None,
        cosine_sim=cosine_sim,
    ).cuda()

    x = torch.randn(2, 16, 32, device="cuda")
    y = attn(x)

    assert y.shape == x.shape
    assert y.is_cuda
    assert torch.isfinite(y).all()


def test_attention_rejects_bad_sequence_length():
    attn = ECGWindowAttention(
        dim=32,
        heads=4,
        window_size=8,
        rpe_type="default",
        shift=False,
        shift_size=None,
        cosine_sim=False,
    ).cuda()

    x = torch.randn(2, 15, 32, device="cuda")

    with pytest.raises(ValueError):
        attn(x)


@pytest.mark.parametrize("rpe_type", [None, "default", "cope", "combined"])
@pytest.mark.parametrize("use_ape", [False, True])
@pytest.mark.parametrize("cosine_sim", [False, True])
@pytest.mark.parametrize("shift", [False, True])
def test_full_model_forward_with_output_head(rpe_type, use_ape, cosine_sim, shift):
    model = _make_model(
        apply_out_mlp=True,
        rpe_type=rpe_type,
        use_ape=use_ape,
        cosine_sim=cosine_sim,
        shift=shift,
    )

    x = torch.randn(2, 256, 12, device="cuda")
    y = model(x)

    assert y.shape == (2, 5)
    assert y.is_cuda
    assert torch.isfinite(y).all()


def test_full_model_forward_without_output_head():
    model = _make_model(apply_out_mlp=False)

    x = torch.randn(2, 256, 12, device="cuda")
    y = model(x)

    assert y.shape == (2, 16, 64)
    assert y.is_cuda
    assert torch.isfinite(y).all()


def test_backward_pass_with_output_head():
    model = _make_model(apply_out_mlp=True)

    x = torch.randn(2, 256, 12, device="cuda")
    target = torch.randn(2, 5, device="cuda")

    y = model(x)
    loss = F.mse_loss(y, target)
    loss.backward()

    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]

    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_backward_pass_without_output_head():
    model = _make_model(apply_out_mlp=False)

    x = torch.randn(2, 256, 12, device="cuda")
    y = model(x)
    loss = y.mean()
    loss.backward()

    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]

    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_conv_merge_output_shape():
    module = ECGConvMerge(
        in_c=12,
        out_c=32,
        seq_len=64,
        downsample_factor=4,
        expansion_factor=2,
        p_drop=0.1,
        conv_config={"kernel_size": 4, "stride": 4, "padding": 0},
    ).cuda()

    x = torch.randn(2, 256, 12, device="cuda")
    y = module(x)

    assert y.shape == (2, 64, 32)
    assert y.is_cuda
    assert torch.isfinite(y).all()


def test_stage_module_output_shape():
    stage = ECGStageModule(
        in_c=12,
        out_c=32,
        downsample_factor=4,
        expansion_factor=2,
        merge_p_drop=0.1,
        merge_conv_config={"kernel_size": 4, "stride": 4, "padding": 0},
        layers=2,
        num_heads=4,
        window_size=8,
        attn_p_drop=0.1,
        rpe_type="default",
        cosine_sim=False,
        shift=True,
        shift_size=None,
        use_ape=True,
        merged_seq_len=64,
    ).cuda()

    x = torch.randn(2, 256, 12, device="cuda")
    y = stage(x)

    assert y.shape == (2, 64, 32)
    assert y.is_cuda
    assert torch.isfinite(y).all()


def test_stage_scalar_params_are_expanded():
    model = _make_model(apply_out_mlp=True)

    assert model.expansion_factor == [2, 2]
    assert model.merge_p_drop == [0.1, 0.1]
    assert model.attn_p_drop == [0.1, 0.1]
    assert model.rpe_type == ["default", "default"]
    assert model.use_ape == [True, True]


def test_dict_stage_param_is_copied_per_stage():
    conv_config = {"kernel_size": 4, "stride": 4, "padding": 0}

    model = ECGHiTNeXt(
        ecg_size=(256, 12),
        num_stages=2,
        hidden_dim=[32, 64],
        layers=[2, 2],
        heads=[4, 4],
        downsample_factor=[4, 4],
        window_size=[8, 4],
        expansion_factor=2,
        merge_conv_config=conv_config,
        merge_p_drop=0.1,
        attn_p_drop=0.1,
        rpe_type="default",
        use_ape=True,
        cosine_sim=False,
        shift=True,
        shift_size=None,
        apply_out_mlp=True,
        out_mlp_hidden_dim=(32,),
        out_mlp_out_dim=5,
        out_mlp_dropout=0.1,
        out_mlp_norm=True,
    ).cuda()

    assert model.merge_conv_config == [conv_config, conv_config]
    assert model.merge_conv_config[0] is not model.merge_conv_config[1]


@pytest.mark.parametrize(
    "bad_name,bad_value",
    [
        ("hidden_dim", [32]),
        ("layers", [2]),
        ("heads", [4]),
        ("downsample_factor", [4]),
        ("window_size", [8]),
        ("merge_conv_config", [{"kernel_size": 4, "stride": 4, "padding": 0}]),
    ],
)
def test_invalid_stage_list_lengths_raise(bad_name, bad_value):
    kwargs = dict(
        ecg_size=(256, 12),
        num_stages=2,
        hidden_dim=[32, 64],
        layers=[2, 2],
        heads=[4, 4],
        downsample_factor=[4, 4],
        window_size=[8, 4],
        expansion_factor=2,
        merge_conv_config=[
            {"kernel_size": 4, "stride": 4, "padding": 0},
            {"kernel_size": 4, "stride": 4, "padding": 0},
        ],
        merge_p_drop=0.1,
        attn_p_drop=0.1,
        rpe_type="default",
        use_ape=True,
        cosine_sim=False,
        shift=True,
        shift_size=None,
        apply_out_mlp=True,
        out_mlp_hidden_dim=(32,),
        out_mlp_out_dim=5,
        out_mlp_dropout=0.1,
        out_mlp_norm=True,
    )

    kwargs[bad_name] = bad_value

    with pytest.raises(ValueError):
        ECGHiTNeXt(**kwargs).cuda()


def test_invalid_odd_number_of_layers_raises():
    with pytest.raises(AssertionError):
        _make_model(apply_out_mlp=True).stages.append(
            ECGStageModule(
                in_c=12,
                out_c=32,
                downsample_factor=4,
                expansion_factor=2,
                merge_p_drop=0.1,
                merge_conv_config={"kernel_size": 4, "stride": 4, "padding": 0},
                layers=3,
                num_heads=4,
                window_size=8,
                attn_p_drop=0.1,
                rpe_type="default",
                cosine_sim=False,
                shift=True,
                shift_size=None,
                use_ape=True,
                merged_seq_len=64,
            ).cuda()
        )


def test_hidden_dim_must_be_divisible_by_heads():
    with pytest.raises(AssertionError):
        ECGHiTNeXt(
            ecg_size=(256, 12),
            num_stages=1,
            hidden_dim=[30],
            layers=[2],
            heads=[4],
            downsample_factor=[4],
            window_size=[8],
            expansion_factor=2,
            merge_conv_config=[{"kernel_size": 4, "stride": 4, "padding": 0}],
            merge_p_drop=0.1,
            attn_p_drop=0.1,
            rpe_type="default",
            use_ape=True,
            cosine_sim=False,
            shift=True,
            shift_size=None,
            apply_out_mlp=True,
            out_mlp_hidden_dim=(32,),
            out_mlp_out_dim=5,
            out_mlp_dropout=0.1,
            out_mlp_norm=True,
        ).cuda()


def test_sequence_length_must_be_divisible_by_downsample_factor():
    with pytest.raises(AssertionError):
        ECGHiTNeXt(
            ecg_size=(250, 12),
            num_stages=1,
            hidden_dim=[32],
            layers=[2],
            heads=[4],
            downsample_factor=[4],
            window_size=[8],
            expansion_factor=2,
            merge_conv_config=[{"kernel_size": 4, "stride": 4, "padding": 0}],
            merge_p_drop=0.1,
            attn_p_drop=0.1,
            rpe_type="default",
            use_ape=True,
            cosine_sim=False,
            shift=True,
            shift_size=None,
            apply_out_mlp=True,
            out_mlp_hidden_dim=(32,),
            out_mlp_out_dim=5,
            out_mlp_dropout=0.1,
            out_mlp_norm=True,
        ).cuda()


def test_merged_sequence_length_must_be_divisible_by_window_size():
    with pytest.raises(AssertionError):
        ECGHiTNeXt(
            ecg_size=(256, 12),
            num_stages=1,
            hidden_dim=[32],
            layers=[2],
            heads=[4],
            downsample_factor=[4],
            window_size=[7],
            expansion_factor=2,
            merge_conv_config=[{"kernel_size": 4, "stride": 4, "padding": 0}],
            merge_p_drop=0.1,
            attn_p_drop=0.1,
            rpe_type="default",
            use_ape=True,
            cosine_sim=False,
            shift=True,
            shift_size=None,
            apply_out_mlp=True,
            out_mlp_hidden_dim=(32,),
            out_mlp_out_dim=5,
            out_mlp_dropout=0.1,
            out_mlp_norm=True,
        ).cuda()


def test_invalid_rpe_type_raises():
    with pytest.raises(ValueError):
        ECGWindowAttention(
            dim=32,
            heads=4,
            window_size=8,
            rpe_type="invalid",
            shift=False,
            shift_size=None,
            cosine_sim=False,
        ).cuda()


def test_relative_indices_buffer_moves_to_gpu():
    attn = ECGWindowAttention(
        dim=32,
        heads=4,
        window_size=8,
        rpe_type="default",
        shift=False,
        shift_size=None,
        cosine_sim=False,
    ).cuda()

    assert attn.relative_indices.is_cuda


def test_grn_initially_identity():
    grn = GRN(dim=32).cuda()
    x = torch.randn(2, 32, 64, device="cuda")

    y = grn(x)

    assert torch.allclose(y, x, atol=1e-6)
