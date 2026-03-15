import torch
import copy
# from main import compute_class_weights, save_metrics
from src.utils.helpers import build_model, compute_class_weights, save_metrics
from src.models.losses import build_loss
from src.engine import train
from src.evaluate import evaluate
from src.utils.visualizer import visualize_tsne_embeddings

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