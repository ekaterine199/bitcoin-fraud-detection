import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  

class FraudGCN(torch.nn.Module): 
    def __init__(self, in_channels=166, hidden_channels=64, out_channels=2, num_layers=3, dropout=0.5):
        
        super(FraudGCN, self).__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels)) 
            
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, return_embeddings=False):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.convs[-1](x, edge_index)
        embeddings = x
        out = self.classifier(embeddings)

        if return_embeddings:
            return out, embeddings
        
        return out