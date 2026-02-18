import pandas as pd
import torch
import os
import  glob
from src.config import RAW_DIR

def find_file(filename):
    """პოულობს ფაილს RAW_DIR-ში ან მის ნებისმიერ ქვე-ფოლდერში"""
    search_path = os.path.join(RAW_DIR, "**", filename)
    files = glob.glob(search_path, recursive=True)
    if not files:
        raise FileNotFoundError(f"❌ ფაილი {filename} ვერ მოიძებნა {RAW_DIR}-ში!")
    return files[0]

def process_raw_data():
    print("⏳ Processing raw CSV files... This might take a while.")
    
    # 1. Load Data
    classes = pd.read_csv(os.path.join(RAW_DIR, 'elliptic_txs_classes.csv'))
    edges = pd.read_csv(os.path.join(RAW_DIR, 'elliptic_txs_edgelist.csv'))
    features = pd.read_csv(os.path.join(RAW_DIR, 'elliptic_txs_features.csv'), header=None)
    
    # 2. Merge & Clean (Elliptic Specifics)
    classes = classes.rename(columns={'txId': 'tx_id'})
    edges = edges.rename(columns={'txId1': 'source', 'txId2': 'target'})
    features = features.rename(columns={0: 'tx_id'})
    
    # 3. Map Classes (Illicit=1, Licit=0, Unknown=-1)
    class_map = {'illicit': 1, 'licit': 0, 'unknown': -1}
    classes['class'] = classes['class'].map(class_map)
    
    # TODO: Add more Elliptic-specific processing if needed (e.g., handling time steps, creating additional features)
    
    data_object = {
        "x": torch.tensor(features.iloc[:, 1:].values, dtype=torch.float), # Features
        "y": torch.tensor(classes['class'].values, dtype=torch.long),      # Labels
        "edge_index": torch.tensor(edges[['source', 'target']].values.T, dtype=torch.long) # Edges
    }
    
    print("✅ Processing complete.")
    return data_object