from dotenv import load_dotenv

load_dotenv()

from scripts.ecg_dataset_statistics import main


if __name__ == "__main__":
    main()
