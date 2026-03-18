import os
import torch
from src.config import PROCESSED_DIR
from src.downloader import download_data_if_needed
from src.processor import process_raw_data

class EllipticDataset:
    # OLD CODE COMMENTED OUT BELOW:
    # def __init__(self):
    #     self.data = self._load_or_process()

    # NEW CODE =====================================================
    def __init__(self, feature_mode="both", split_mode="strict", include_unknown_class=False):
        self.feature_mode = feature_mode
        self.split_mode = split_mode
        self.include_unknown_class = include_unknown_class
        self.data = self._load_or_process()
    # =============================================================

    def _load_or_process(self):
        # OLD CODE COMMENTED OUT BELOW:
        # if os.path.exists(PROCESSED_FILE):
        #     print(f"🚀 Loading cached graph data from {PROCESSED_FILE}")
        #     return torch.load(PROCESSED_FILE, weights_only=False)

        # NEW CODE =====================================================
        processed_file = os.path.join(
            PROCESSED_DIR,
            f"elliptic_graph_{self.feature_mode}_{self.split_mode}_{'with_unknown' if self.include_unknown_class else 'known_only'}.pt"
        )

        if os.path.exists(processed_file):
            print(f"🚀 Loading cached graph data from {processed_file}")
            return torch.load(processed_file, weights_only=False)
        # =============================================================
        
        print("⚠️ Processed data missing. Initializing setup sequence...")

        download_data_if_needed()

        # OLD CODE COMMENTED OUT BELOW:
        # data = process_raw_data()

        # NEW CODE =====================================================
        data = process_raw_data(
            feature_mode=self.feature_mode,
            split_mode=self.split_mode,
            include_unknown_class=self.include_unknown_class
        )
        # =============================================================
        
        # OLD CODE COMMENTED OUT BELOW:
        # torch.save(data, PROCESSED_FILE)
        # print(f"💾 Processed data saved to {PROCESSED_FILE}")

        # NEW CODE =====================================================
        torch.save(data, processed_file)
        print(f"💾 Processed data saved to {processed_file}")
        # =============================================================
        
        return data