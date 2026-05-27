from dotenv import load_dotenv

load_dotenv()

from scripts import ecg_hit_next_train

if __name__ == "__main__":
    ecg_hit_next_train()
