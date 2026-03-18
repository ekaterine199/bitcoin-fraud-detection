import torch
import copy
from src.utils.helpers import build_model, compute_class_weights, save_metrics, build_optimizer
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


# NEW CODE =====================================================
def select_feature_mode_from_menu():
    print("\nSelect feature subset:")
    print("[1] Both regular + aggregated")
    print("[2] Regular only")
    print("[3] Aggregated only")

    options = {
        "1": "both",
        "2": "regular",
        "3": "aggregated"
    }

    while True:
        choice = input("Enter feature option: ").strip()
        if choice in options:
            return options[choice]
        print("Invalid selection. Please enter 1, 2, or 3.")


def select_split_mode_from_menu():
    print("\nSelect timestep split mode:")
    print("[1] strict")
    print("[2] 70_30")
    print("[3] 75_25")
    print("[4] 80_20")

    options = {
        "1": "strict",
        "2": "70_30",
        "3": "75_25",
        "4": "80_20"
    }

    while True:
        choice = input("Enter split option: ").strip()
        if choice in options:
            return options[choice]
        print("Invalid selection. Please enter 1, 2, 3, or 4.")


def select_unknown_mode_from_menu():
    print("\nUnknown class handling:")
    print("[1] Exclude unknown from supervision")
    print("[2] Include unknown as class 2")

    options = {
        "1": False,
        "2": True
    }

    while True:
        choice = input("Enter unknown option: ").strip()
        if choice in options:
            return options[choice]
        print("Invalid selection. Please enter 1 or 2.")


def select_optimizer_from_menu():
    print("\nSelect optimizer:")
    print("[1] Adam")
    print("[2] AdamW")
    print("[3] SGD")
    print("[4] RMSprop")

    options = {
        "1": "adam",
        "2": "adamw",
        "3": "sgd",
        "4": "rmsprop"
    }

    while True:
        choice = input("Enter optimizer number: ").strip()
        if choice in options:
            return options[choice]
        print("Invalid selection. Please enter 1, 2, 3, or 4.")


def select_learning_rate_from_menu():
    print("\nSelect learning rate:")
    print("[1] 0.01")
    print("[2] 0.005")
    print("[3] 0.001")
    print("[4] 0.0005")

    options = {
        "1": 1e-2,
        "2": 5e-3,
        "3": 1e-3,
        "4": 5e-4
    }

    while True:
        choice = input("Enter learning rate number: ").strip()
        if choice in options:
            return options[choice]
        print("Invalid selection. Please enter 1, 2, 3, or 4.")
# =============================================================


def ask_run_again():
    while True:
        choice = input("\nRun another experiment? [y/n]: ").strip().lower()
        if choice in ["y", "yes"]:
            return True
        if choice in ["n", "no"]:
            return False
        print("Invalid selection. Please enter y or n.")

# OLD CODE COMMENTED OUT BELOW:
# def run_single_experiment(data, device, model_name, loss_name, run_id):
def run_single_experiment(data, device, model_name, loss_name, optimizer_name, learning_rate, run_id):
    model = build_model(model_name, data).to(device)

    class_weights = compute_class_weights(data).to(device)
    criterion = build_loss(loss_name, class_weights)

    # OLD CODE COMMENTED OUT BELOW:
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # NEW CODE =====================================================
    optimizer = build_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        learning_rate=learning_rate,
        weight_decay=5e-4
    )
    # =============================================================

    print(f"\n🧠 Model running on: {device}")
    # OLD CODE COMMENTED OUT BELOW:
    # print(f"📌 Model: {model_name.upper()} | Loss: {loss_name} | Run: {run_id}")

    # NEW CODE =====================================================
    print(f"📌 Model: {model_name.upper()} | Loss: {loss_name} | Optimizer: {optimizer_name} | LR: {learning_rate} | Run: {run_id}")
    # =============================================================
    print(f"⚖️ Class weights: {class_weights.detach().cpu().tolist()}")

    # Early stopping based on validation illicit F1.
    # OLD CODE COMMENTED OUT BELOW:
    # best_val_f1 = -1.0

    # NEW CODE =====================================================
    monitor_key = "macro_f1_all" if getattr(data, "include_unknown_class", False) else "f1_illicit"
    best_val_score = -1.0
    # =============================================================

    best_state = None
    patience = 100
    patience_counter = 0

    for epoch in range(1, 201):
        loss = train(model, data, optimizer, criterion, device)

        val_metrics = evaluate(model, data, data.val_mask)

        # OLD CODE COMMENTED OUT BELOW:
        # if val_metrics["f1_illicit"] > best_val_f1:
        #     best_val_f1 = val_metrics["f1_illicit"]

        # NEW CODE =====================================================
        if val_metrics[monitor_key] > best_val_score:
            best_val_score = val_metrics[monitor_key]
        # =============================================================
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

        # NEW CODE =====================================================
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "feature_mode": getattr(data, "feature_mode", "unknown"),
        "split_mode": getattr(data, "split_mode", "unknown"),
        "include_unknown_class": bool(getattr(data, "include_unknown_class", False)),
        # =============================================================

        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics
    }

    # OLD CODE COMMENTED OUT BELOW:
    # metrics_file = f"{model_name}_{loss_name}_run{run_id}_metrics.json"
    # tsne_file = f"tsne_{model_name}_{loss_name}_run{run_id}.png"

    # NEW CODE =====================================================
    metrics_file = (
        f"{model_name}_{loss_name}_{optimizer_name}_lr{learning_rate}_"
        f"{data.feature_mode}_{data.split_mode}_"
        f"{'with_unknown' if data.include_unknown_class else 'known_only'}_run{run_id}_metrics.json"
    )
    tsne_file = (
        f"tsne_{model_name}_{loss_name}_{optimizer_name}_lr{learning_rate}_"
        f"{data.feature_mode}_{data.split_mode}_"
        f"{'with_unknown' if data.include_unknown_class else 'known_only'}_run{run_id}.png"
    )
    # =============================================================

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