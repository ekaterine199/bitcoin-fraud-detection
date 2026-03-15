from src.models.gcn import FraudGCN
from src.models.sage import FraudGraphSAGE
from src.models.gat import GAT

def build_model(model_name, data):
    if model_name == "gcn":
        return FraudGCN(
            in_channels=data.x.shape[1],
            hidden_channels=128,
            out_channels=2,
            num_layers=3,
            dropout=0.5
        )

    if model_name == "sage":
        return FraudGraphSAGE(
            in_channels=data.x.shape[1],
            hidden_channels=128,
            out_channels=2,
            num_layers=3,
            dropout=0.5
        )

    if model_name == "gat":
        return GAT(
            in_channels=data.x.shape[1],
            hidden_channels=32,
            out_channels=2,
            heads=4,
            dropout=0.4
        )

    raise ValueError(f"Unsupported model_name: {model_name}")
