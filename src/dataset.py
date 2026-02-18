import os
import torch
from src.config import PROCESSED_FILE
from src.downloader import download_data_if_needed
from src.processor import process_raw_data

class EllipticDataset:
    def __init__(self):
        self.data = self._load_or_process()

    def _load_or_process(self):
        if os.path.exists(PROCESSED_FILE):
            print(f"🚀 Loading cached graph data from {PROCESSED_FILE}")
            return torch.load(PROCESSED_FILE)
        
        print("⚠️ Processed data missing. Initializing setup sequence...")

        download_data_if_needed()

        data = process_raw_data()
        
        torch.save(data, PROCESSED_FILE)
        print(f"💾 Processed data saved to {PROCESSED_FILE}")
        
        return data