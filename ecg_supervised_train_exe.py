from dotenv import load_dotenv

load_dotenv()

from scripts import ecg_supervised_train

if __name__ == "__main__":
    ecg_supervised_train()
