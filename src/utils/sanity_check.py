import torch

def perform_sanity_check(data):
    print("🔍 Running sanity check on the graph data...")
    
    assert data.x is not None, "Data node features (x) are missing!"
    assert data.edge_index is not None, "Edge index is missing!"
    
    assert not torch.isnan(data.x).any(), "Found NaN in features (x)!"
    assert not torch.isinf(data.x).any(), "Found Inf in features (x)!"
    
    num_nodes = data.x.shape[0]
    if data.edge_index.numel() > 0:
        assert data.edge_index.max() < num_nodes, "Edge index out of bounds (max index > num_nodes)!"
        assert data.edge_index.min() >= 0, "Edge index contains negative values!"
    
    if hasattr(data, 'y') and data.y is not None:
        unique_labels = torch.unique(data.y)
        print(f"   - Detected unique labels: {unique_labels.tolist()}")

    if hasattr(data, 'train_mask'):
        assert data.train_mask.shape[0] == num_nodes, "train_mask size does not match number of nodes!"
        print(f"   - Train labeled nodes: {int(data.train_mask.sum())}")

    if hasattr(data, 'val_mask'):
        assert data.val_mask.shape[0] == num_nodes, "val_mask size does not match number of nodes!"
        print(f"   - Val labeled nodes:   {int(data.val_mask.sum())}")

    if hasattr(data, 'test_mask'):
        assert data.test_mask.shape[0] == num_nodes, "test_mask size does not match number of nodes!"
        print(f"   - Test labeled nodes:  {int(data.test_mask.sum())}")

    if hasattr(data, 'y') and data.y is not None:
        unknown_count = int((data.y == -1).sum())
        print(f"   - Unknown nodes:       {unknown_count}")
        
    print("✅ Sanity check passed successfully!")