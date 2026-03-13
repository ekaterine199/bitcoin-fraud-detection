# gat.py

# =====================================================================
# Graph Attention Network (GAT) model for the Elliptic Bitcoin dataset.
# =====================================================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.4):
        super().__init__()

        self.dropout = dropout

        # First GAT layer with multi-head attention
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout
        )

        # Second GAT layer outputs hidden embeddings
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            concat=True,
            dropout=dropout
        )

        # Final classifier
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Save final hidden layer embeddings for t-SNE visualization
        embeddings = x

        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin(x)

        if return_embeddings:
            return out, embeddings

        return out