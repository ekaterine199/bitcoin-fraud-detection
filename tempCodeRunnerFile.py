import torch
from torch_geometric.data import Data
from src.dataset import EllipticDataset
from src.models.sage import FraudGraphSAGE


def main():
    print("🚀 Starting the pipeline...")

    dataset = EllipticDataset()

    raw_data = dataset.data  
    
    if isinstance(raw_data, dict):
        data = Data(**raw_data)
    else:
        data = raw_data
        
    print("✅ Graph Data is ready!")
    print(f"Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}, Features: {data.x.shape[1]}")
    
    # 2. Initialize the GraphSAGE Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FraudGraphSAGE(
        in_channels=data.x.shape[1], 
        hidden_channels=128,         
        out_channels=2,              
        num_layers=3                 
    ).to(device)
    
    data = data.to(device)
    
    print(f"🧠 Model initialized on {device}:\n", model)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
    print("🎯 Output tensor shape:", out.shape) 
    print("Pipeline check complete. Ready to build the training loop!")

if __name__ == "__main__":
    main()