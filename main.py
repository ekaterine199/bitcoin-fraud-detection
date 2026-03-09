# import torch
# from torch_geometric.data import Data
# from src.dataset import EllipticDataset
# from src.models.sage import FraudGraphSAGE


# def main():
#     print("🚀 Starting the pipeline...")

#     dataset = EllipticDataset()

#     raw_data = dataset.data  
    
#     if isinstance(raw_data, dict):
#         data = Data(**raw_data)
#     else:
#         data = raw_data
        
#     print("✅ Graph Data is ready!")
#     print(f"Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}, Features: {data.x.shape[1]}")
    
#     # 2. Initialize the GraphSAGE Model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = FraudGraphSAGE(
#         in_channels=data.x.shape[1], 
#         hidden_channels=128,         
#         out_channels=2,              
#         num_layers=3                 
#     ).to(device)
    
#     data = data.to(device)
    
#     print(f"🧠 Model initialized on {device}:\n", model)
    
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x, data.edge_index)
        
#     print("🎯 Output tensor shape:", out.shape) 
#     print("Pipeline check complete. Ready to build the training loop!")

# if __name__ == "__main__":
#     main()



import torch
from torch_geometric.data import Data
from src.models.gcn import FraudGCN
from src.dataset import EllipticDataset
from src.models.sage import FraudGraphSAGE
from src.engine import train
from src.evaluate import evaluate

def main():
    print("🚀 Starting the pipeline...")
    dataset = EllipticDataset()
    raw_data = dataset.data  
    data = Data(**raw_data) if isinstance(raw_data, dict) else raw_data
    
    # მასკების შექმნა, რადგან Elliptic-ს არ აქვს ისინი თავისით
    num_nodes = data.x.shape[0]
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[indices[:train_size]] = True
    
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[indices[train_size:]] = True
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FraudGraphSAGE(
        in_channels=data.x.shape[1], 
        hidden_channels=128,         
        out_channels=2,              
        num_layers=3                 
    ).to(device)
    
    data = data.to(device)
    

    train_labels = data.y[data.train_mask]
    known_mask = (train_labels != -1)

    class_counts = torch.bincount(train_labels[known_mask].long()).float()

    weights = 1.0 / (class_counts + 1e-6)
    weights /= weights.sum()
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"🧠 Model running on: {device}")
    
    for epoch in range(101):
        loss = train(model, data, optimizer, criterion, device)
        
        if epoch % 10 == 0:
            metrics = evaluate(model, data, data.val_mask)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | F1: {metrics['f1']:.4f} | "
                  f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")

    print("✅ Pipeline complete.")

if __name__ == "__main__":
    main()