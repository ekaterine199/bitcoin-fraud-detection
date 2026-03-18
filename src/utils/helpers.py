import torch
import json
from src.models.gcn import FraudGCN
from src.models.sage import FraudGraphSAGE
from src.models.gat import GAT


def build_model(model_name, data):
    # OLD CODE COMMENTED OUT BELOW:
    # out_channels=2

    # NEW CODE =====================================================
    out_channels = int(getattr(data, "num_classes", 2))
    # =============================================================

    if model_name == "gcn":
        return FraudGCN(
            in_channels=data.x.shape[1],
            hidden_channels=128,
            out_channels=out_channels,
            num_layers=3,
            dropout=0.5
        )

    if model_name == "sage":
        return FraudGraphSAGE(
            in_channels=data.x.shape[1],
            hidden_channels=128,
            out_channels=out_channels,
            num_layers=3,
            dropout=0.5
        )

    if model_name == "gat":
        return GAT(
            in_channels=data.x.shape[1],
            hidden_channels=32,
            out_channels=out_channels,
            heads=4,
            dropout=0.4
        )

    raise ValueError(f"Unsupported model_name: {model_name}")

def compute_class_weights(data):
    # OLD CODE COMMENTED OUT BELOW:
    # train_labels = data.y[data.train_mask]
    # known_mask = (train_labels != -1)
    # known_labels = train_labels[known_mask].long()
    # class_counts = torch.bincount(known_labels, minlength=2).float()

    # NEW CODE =====================================================
    supervised_mask = getattr(data, "supervised_train_mask", data.train_mask & (data.y != -1))
    train_labels = data.y[supervised_mask]
    known_labels = train_labels[train_labels != -1].long()
    num_classes = int(getattr(data, "num_classes", 2))
    class_counts = torch.bincount(known_labels, minlength=num_classes).float()
    # =============================================================

    weights = class_counts.sum() / (len(class_counts) * class_counts)
    return weights


# NEW CODE =====================================================
def build_optimizer(optimizer_name, model, learning_rate, weight_decay=5e-4):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
# =============================================================


def save_metrics(metrics_by_split, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_split, f, indent=2)