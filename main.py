# import torch
# from torch_geometric.data import Data
# from src.models.gcn import FraudGCN
# from src.dataset import EllipticDataset
# from src.models.sage import FraudGraphSAGE
# from src.engine import train
# from src.evaluate import evaluate
from src.baseline import run_baseline

def main():
    print("🚀 Starting the pipeline...")
    #dataset = EllipticDataset()
    #raw_data = dataset.data  
    #data = Data(**raw_data) if isinstance(raw_data, dict) else raw_data

    run_baseline()

    # Temporal train/val split — sort by time_step (feature col 0)
    # so the GNN is evaluated the same way as the baseline
    #time_steps = data.x[:, 0].cpu()
    # sorted_idx = torch.argsort(time_steps)
    # num_nodes  = data.x.shape[0]
    # split      = int(0.80 * num_nodes)
    # data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # data.train_mask[sorted_idx[:split]] = True
    # data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # data.val_mask[sorted_idx[split:]] = True
    
    # num_nodes = data.x.shape[0]
    # indices = torch.randperm(num_nodes)
    # train_size = int(0.8 * num_nodes)
    
    # data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # data.train_mask[indices[:train_size]] = True
    
    # data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # data.val_mask[indices[train_size:]] = True
        
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model = FraudGraphSAGE(
    #     in_channels=data.x.shape[1], 
    #     hidden_channels=128,         
    #     out_channels=2,              
    #     num_layers=3                 
    # ).to(device)
    
    # data = data.to(device)
    

    # train_labels = data.y[data.train_mask]
    # known_mask = (train_labels != -1)

    # class_counts = torch.bincount(train_labels[known_mask].long()).float()

    # weights = 1.0 / (class_counts + 1e-6)
    # weights /= weights.sum()
    
    # criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # print(f"🧠 Model running on: {device}")
    
    # for epoch in range(101):
    #     loss = train(model, data, optimizer, criterion, device)
        
    #     if epoch % 10 == 0:
    #         metrics = evaluate(model, data, data.val_mask)
    #         print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | F1: {metrics['f1']:.4f} | "
    #               f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")

    print("✅ Pipeline complete.")

if __name__ == "__main__":
    main()