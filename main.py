import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from src.dataset import EllipticDataset
from src.utils.sanity_check import perform_sanity_check
from utils.cli import select_model_from_menu, select_loss_from_menu, ask_run_again, run_single_experiment
from src.utils.visualizer import visualize_graph
from src.baseline import run_baseline

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