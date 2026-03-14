import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.models.gcn import FraudGCN
from src.models.sage import FraudGraphSAGE
from src.models.gat import GAT
from src.dataset import EllipticDataset
from src.engine import train
from src.evaluate import evaluate
from src.utils.sanity_check import perform_sanity_check

from src.utils.visualizer import visualize_graph, visualize_tsne_embeddings


# NEW CODE =====================================================
# Focal loss option for heavy class imbalance.
# Weighted Cross-Entropy is still the default recommendation.
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


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


def build_loss(loss_name, class_weights):
    if loss_name == "weighted_ce":
        return torch.nn.CrossEntropyLoss(weight=class_weights)

    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=2.0)

    raise ValueError(f"Unsupported loss_name: {loss_name}")


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


def select_model_from_menu():
    print("\nSelect a model to run:")
    print("[1] GCN")
    print("[2] GAT")
    print("[3] GraphSAGE")

    model_options = {
        "1": "gcn",
        "2": "gat",
        "3": "sage"
    }

    while True:
        choice = input("Enter model number: ").strip()
        if choice in model_options:
            return model_options[choice]
        print("Invalid selection. Please enter 1, 2, or 3.")


def select_loss_from_menu():
    print("\nSelect a loss function:")
    print("[1] Weighted Cross-Entropy")
    print("[2] Focal Loss")

    loss_options = {
        "1": "weighted_ce",
        "2": "focal"
    }

    while True:
        choice = input("Enter loss number: ").strip()
        if choice in loss_options:
            return loss_options[choice]
        print("Invalid selection. Please enter 1 or 2.")


def ask_run_again():
    while True:
        choice = input("\nRun another experiment? [y/n]: ").strip().lower()
        if choice in ["y", "yes"]:
            return True
        if choice in ["n", "no"]:
            return False
        print("Invalid selection. Please enter y or n.")
# =============================================================


# NEW CODE =====================================================
def run_single_experiment(data, device, model_name, loss_name, run_id):
    model = build_model(model_name, data).to(device)

    class_weights = compute_class_weights(data).to(device)
    criterion = build_loss(loss_name, class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    print(f"\n🧠 Model running on: {device}")
    print(f"📌 Model: {model_name.upper()} | Loss: {loss_name} | Run: {run_id}")
    print(f"⚖️ Class weights: {class_weights.detach().cpu().tolist()}")

    # Early stopping based on validation illicit F1.
    best_val_f1 = -1.0
    best_state = None
    patience = 100
    patience_counter = 0

    for epoch in range(1, 201):
        loss = train(model, data, optimizer, criterion, device)

        val_metrics = evaluate(model, data, data.val_mask)

        if val_metrics["f1_illicit"] > best_val_f1:
            best_val_f1 = val_metrics["f1_illicit"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            train_metrics = evaluate(model, data, data.train_mask)

            print(
                f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                f"Train F1(illicit): {train_metrics['f1_illicit']:.4f} | "
                f"Val F1(illicit): {val_metrics['f1_illicit']:.4f} | "
                f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                f"Val Precision(illicit): {val_metrics['precision_illicit']:.4f} | "
                f"Val Recall(illicit): {val_metrics['recall_illicit']:.4f}"
            )

        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, data, data.train_mask)
    val_metrics = evaluate(model, data, data.val_mask)
    test_metrics = evaluate(model, data, data.test_mask)

    print("\nFinal metrics")
    print(f"Train: {train_metrics}")
    print(f"Val:   {val_metrics}")
    print(f"Test:  {test_metrics}")

    metrics_by_split = {
        "run_id": run_id,
        "model_name": model_name,
        "loss_name": loss_name,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics
    }

    metrics_file = f"{model_name}_{loss_name}_run{run_id}_metrics.json"
    tsne_file = f"tsne_{model_name}_{loss_name}_run{run_id}.png"

    save_metrics(metrics_by_split, metrics_file)

    # Generate t-SNE using the final hidden layer embeddings.
    # This requires the model to support return_embeddings=True.
    visualize_tsne_embeddings(
        model=model,
        data=data,
        mask=data.test_mask,
        output_dir="img",
        file_name=tsne_file
    )

    print(f"✅ Saved metrics to {metrics_file}")
    print(f"✅ Saved t-SNE to img/{tsne_file}")

    return metrics_by_split

from src.utils.visualizer import visualize_graph
from src.baseline import run_baseline

def main():
    print("🚀 Starting the pipeline...")
    dataset = EllipticDataset()
    raw_data = dataset.data
    data = Data(**raw_data) if isinstance(raw_data, dict) else raw_data

    run_baseline()    
    perform_sanity_check(data)
    visualize_graph(data, output_dir="img", num_nodes=200)

    if not hasattr(data, "train_mask") or not hasattr(data, "val_mask") or not hasattr(data, "test_mask"):
        raise ValueError("Data object is missing train/val/test masks. Check processor.py.")

    print(f"Train labeled nodes: {int(data.train_mask.sum())}")
    print(f"Val labeled nodes:   {int(data.val_mask.sum())}")
    print(f"Test labeled nodes:  {int(data.test_mask.sum())}")
    print(f"Unknown nodes:       {int((data.y == -1).sum())}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    session_results = []
    run_id = 1

    while True:
        model_name = select_model_from_menu()
        loss_name = select_loss_from_menu()

        experiment_result = run_single_experiment(
            data=data,
            device=device,
            model_name=model_name,
            loss_name=loss_name,
            run_id=run_id
        )

        session_results.append(experiment_result)
        run_id += 1

        if not ask_run_again():
            break

    save_metrics(session_results, "experiment_session_summary.json")

    print("✅ Pipeline complete.")


if __name__ == "__main__":
    main()