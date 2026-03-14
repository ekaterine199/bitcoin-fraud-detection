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
        
    print("✅ Sanity check passed successfully!")