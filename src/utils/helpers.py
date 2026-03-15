import torch
import json
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

def compute_class_weights(data):
    train_labels = data.y[data.train_mask]
    known_mask = (train_labels != -1)
    known_labels = train_labels[known_mask].long()

    class_counts = torch.bincount(known_labels, minlength=2).float()

    # Inverse-frequency style weighting
    weights = class_counts.sum() / (2.0 * class_counts)
    return weights


def save_metrics(metrics_by_split, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_split, f, indent=2)