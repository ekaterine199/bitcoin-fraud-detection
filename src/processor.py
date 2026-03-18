import pandas as pd
import numpy as np
import torch
import os
import glob
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from src.config import RAW_DIR

def find_file(filename):
    search_path = os.path.join(RAW_DIR, "**", filename)
    files = glob.glob(search_path, recursive=True)
    if not files:
        raise FileNotFoundError(f"❌ File {filename} not found in {RAW_DIR} !")
    return files[0]


# NEW CODE =====================================================
def _resolve_feature_matrix(full_feature_matrix, feature_mode):
    """
    Feature layout after dropping tx_id and class:
      col 0   -> timestep
      cols 0:94   -> regular/local block
      cols 94:166 -> aggregated block
    """
    if feature_mode == "both":
        return full_feature_matrix

    if feature_mode == "regular":
        return full_feature_matrix[:, :94]

    if feature_mode == "aggregated":
        return full_feature_matrix[:, 94:]

    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def _resolve_split_masks(timesteps, base_mask, split_mode):
    """
    Returns train/val/test masks based on timestep.
    strict keeps the project split exactly.
    The ratio-based modes keep the last few train timesteps for validation.
    """
    max_ts = int(timesteps.max().item())

    # OLD CODE COMMENTED OUT BELOW:
    # train_mask = (timesteps >= 1) & (timesteps <= 30) & labeled_mask
    # val_mask = (timesteps >= 31) & (timesteps <= 34) & labeled_mask
    # test_mask = (timesteps >= 35) & (timesteps <= 49) & labeled_mask

    if split_mode == "strict":
        train_mask = (timesteps >= 1) & (timesteps <= 30) & base_mask
        val_mask = (timesteps >= 31) & (timesteps <= 34) & base_mask
        test_mask = (timesteps >= 35) & (timesteps <= 49) & base_mask
        return train_mask, val_mask, test_mask

    if split_mode == "70_30":
        split_end = int(round(max_ts * 0.70))
    elif split_mode == "75_25":
        split_end = int(round(max_ts * 0.75))
    elif split_mode == "80_20":
        split_end = int(round(max_ts * 0.80))
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    val_width = 4
    val_start = max(2, split_end - val_width + 1)

    train_mask = (timesteps >= 1) & (timesteps <= val_start - 1) & base_mask
    val_mask = (timesteps >= val_start) & (timesteps <= split_end) & base_mask
    test_mask = (timesteps >= split_end + 1) & (timesteps <= max_ts) & base_mask

    return train_mask, val_mask, test_mask
# =============================================================


# OLD CODE COMMENTED OUT BELOW:
# def process_raw_data():
def process_raw_data(feature_mode="both", split_mode="strict", include_unknown_class=False):
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
    # OLD CODE COMMENTED OUT BELOW:
    # class_map = {"1": 1, "2": 0, "unknown": -1}
    # combined_df['class'] = combined_df['class'].astype(str).map(class_map)

    # NEW CODE =====================================================
    if include_unknown_class:
        class_map = {"1": 1, "2": 0, "unknown": 2}
        num_classes = 3
    else:
        class_map = {"1": 1, "2": 0, "unknown": -1}
        num_classes = 2

    combined_df['class'] = combined_df['class'].astype(str).map(class_map)
    # =============================================================

    timestep_col = combined_df.columns[1]
    combined_df.rename(columns={timestep_col: 'timestep'}, inplace=True)
    
    # 5. Prepare Tensors
    # x = torch.tensor(combined_df.drop(columns=['tx_id', 'class']).values, dtype=torch.float)
    # OLD CODE COMMENTED OUT BELOW:
    # x_np = combined_df.drop(columns=['tx_id', 'class']).values
    x_np_full = combined_df.drop(columns=['tx_id', 'class']).values

    # NEW CODE =====================================================
    x_np = _resolve_feature_matrix(x_np_full, feature_mode)
    # =============================================================

    y = torch.tensor(combined_df['class'].values, dtype=torch.long)
    timesteps = torch.tensor(combined_df['timestep'].values, dtype=torch.long)
    
    # 6. Transform Edge Index 
    edges_df['source'] = edges_df['txId1'].map(id_map)
    edges_df['target'] = edges_df['txId2'].map(id_map)
    
    edges_df.dropna(subset=['source', 'target'], inplace=True)
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)

    # OLD CODE COMMENTED OUT BELOW:
    # labeled_mask = (y != -1)
    # train_mask = (timesteps >= 1) & (timesteps <= 30) & labeled_mask
    # val_mask = (timesteps >= 31) & (timesteps <= 34) & labeled_mask
    # test_mask = (timesteps >= 35) & (timesteps <= 49) & labeled_mask

    # NEW CODE =====================================================
    if include_unknown_class:
        base_mask = torch.ones_like(y, dtype=torch.bool)
    else:
        base_mask = (y != -1)

    train_mask, val_mask, test_mask = _resolve_split_masks(
        timesteps=timesteps,
        base_mask=base_mask,
        split_mode=split_mode
    )

    supervised_train_mask = train_mask if include_unknown_class else (train_mask & (y != -1))
    # =============================================================

    scaler = StandardScaler()
    # scaler.fit(x_np[train_mask.numpy()])
    # x_scaled = scaler.transform(x_np)
    # x = torch.tensor(x_scaled, dtype=torch.float)
    time_feature   = x_np[:, 0:1]                    
    other_features = x_np[:, 1:]                     

    # OLD CODE COMMENTED OUT BELOW:
    # scaler.fit(other_features[train_mask.numpy()])   

    # NEW CODE =====================================================
    scaler.fit(other_features[supervised_train_mask.numpy()])   
    # =============================================================

    other_scaled = scaler.transform(other_features)
    x_scaled = np.hstack([time_feature, other_scaled])   

    x = torch.tensor(x_scaled, dtype=torch.float)
    
    data_object = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        timestep=timesteps,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    data_object.id_map = id_map

    # NEW CODE =====================================================
    data_object.feature_mode = feature_mode
    data_object.split_mode = split_mode
    data_object.include_unknown_class = include_unknown_class
    data_object.num_classes = num_classes
    data_object.supervised_train_mask = supervised_train_mask
    # =============================================================

    print(f"✅ Processing complete. Nodes: {x.shape[0]}, Edges: {edge_index.shape[1]}")
    print(f"   - Feature mode:        {feature_mode}")
    print(f"   - Split mode:          {split_mode}")
    print(f"   - Include unknown:     {include_unknown_class}")
    print(f"   - Train labeled nodes: {int(train_mask.sum())}")
    print(f"   - Val labeled nodes:   {int(val_mask.sum())}")
    print(f"   - Test labeled nodes:  {int(test_mask.sum())}")
    if include_unknown_class:
        print(f"   - Unknown nodes:       {int((y == 2).sum())}")
    else:
        print(f"   - Unknown nodes:       {int((y == -1).sum())}")
    return data_object