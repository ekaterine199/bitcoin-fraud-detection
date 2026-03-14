import os
import shutil
import glob
from src.config import DATASET_NAME, RAW_DIR, RAW_FILE_CHECK

def download_data_if_needed():
    # Check if raw data already exists
    if os.path.exists(RAW_FILE_CHECK):
        print("✅ Raw data already exists. Skipping download.")
        return

    print(f"⬇️ Raw data not found. Downloading {DATASET_NAME} from Kaggle...")
    
    try:
        # OLD CODE COMMENTED OUT BELOW:
        # from kaggle.api.kaggle_api_extended import KaggleApi

        # NEW CODE =====================================================
        # Import Kaggle only when a download is actually needed.
        # This prevents ModuleNotFoundError when the raw CSVs are already present.
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Allow kaggle.json to be placed in the project root.
        os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
        # =============================================================

        # Download and unzip the dataset using Kaggle API
        api = KaggleApi()
        api.authenticate() 
        
        # Download and extract the dataset to the RAW_DIR
        api.dataset_download_files(DATASET_NAME, path=RAW_DIR, unzip=True)

        extracted_csvs = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)

        for file_path in extracted_csvs:
            filename = os.path.basename(file_path)
            destination_path = os.path.join(RAW_DIR, filename)

            # Move the file to RAW_DIR if it's not already there
            if os.path.abspath(file_path) != os.path.abspath(destination_path):
                shutil.move(file_path, destination_path)
                print(f"🚚 Moved: {filename} to data/raw/")

        # 5. Delete any remaining folders in RAW_DIR
        for item in os.listdir(RAW_DIR):
            item_path = os.path.join(RAW_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"🧹 Cleaned up folder: {item}")
        
        print("✅ Download and extraction complete.")
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")

        # OLD CODE COMMENTED OUT BELOW:
        # print("⚠️ Ensure 'kaggle.json' is in the project root folder!")

        # NEW CODE =====================================================
        print("⚠️ Ensure 'kaggle.json' is in the project root folder or that the raw CSVs already exist in data/raw.")
        # =============================================================
        raise e