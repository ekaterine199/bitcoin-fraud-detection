import pandas as pd
import torch
import os
import  glob
from src.config import RAW_DIR

def find_file(filename):
    search_path = os.path.join(RAW_DIR, "**", filename)
    files = glob.glob(search_path, recursive=True)
    if not files:
        raise FileNotFoundError(f"❌ File {filename} not found in {RAW_DIR} !")
    return files[0]

def process_raw_data():
    print("⏳ Processing raw CSV files... This might take a while.")
    
    # 1. Load Data
    classes_df = pd.read_csv(os.path.join(RAW_DIR, 'elliptic_txs_classes.csv'))
    edges_df = pd.read_csv(os.path.join(RAW_DIR, 'elliptic_txs_edgelist.csv'))
    features_df = pd.read_csv(os.path.join(RAW_DIR, 'elliptic_txs_features.csv'), header=None)
    
    # Change column names for consistency
    classes_df.rename(columns={'txId': 'tx_id'}, inplace=True)
    features_df.rename(columns={0: 'tx_id'}, inplace=True)

    # 2. Node ID Mapping 
    # Create dictionary to map original tx_id to a contiguous range of integers
    all_tx_ids = features_df['tx_id'].unique()
    id_map = {old_id: new_idx for new_idx, old_id in enumerate(all_tx_ids)}
    
    # 3. Features & Labels Alignment
    combined_df = pd.merge(features_df, classes_df, on='tx_id', how='left')
    
    # 4. Map Classes 
    class_map = {"1": 1, "2": 0, "unknown": -1}
    combined_df['class'] = combined_df['class'].map(class_map)
    
    # 5. Prepare Tensors
    x = torch.tensor(combined_df.drop(columns=['tx_id', 'class']).values, dtype=torch.float)
    y = torch.tensor(combined_df['class'].values, dtype=torch.long)
    
    # 6. Transform Edge Index 
    edges_df['source'] = edges_df['txId1'].map(id_map)
    edges_df['target'] = edges_df['txId2'].map(id_map)
    
    edges_df.dropna(subset=['source', 'target'], inplace=True)
    
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)
    
    data_object = {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "id_map": id_map  
    }
    
    print(f"✅ Processing complete. Nodes: {x.shape[0]}, Edges: {edge_index.shape[1]}")
    return data_object