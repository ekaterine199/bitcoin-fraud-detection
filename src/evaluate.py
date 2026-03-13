# src/evaluate.py

import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score
)

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        # ამოვიღოთ ნამდვილი და ნაწინასწარმეტყველები მონაცემები

        # OLD CODE COMMENTED OUT BELOW:
        # y_true = data.y[mask]
        # y_pred = pred[mask]

        # NEW CODE =====================================================
        # Filter to nodes in the provided mask, then remove unknown labels.
        y_true = data.y[mask]
        y_pred = pred[mask]
        y_score = torch.softmax(out[mask], dim=1)[:, 1]
        # ==============================================================

        # OLD CODE COMMENTED OUT BELOW:
        # valid_mask = (y_true != -1)

        # გაფილტვრა -1-ებისგან

        valid_mask = (y_true != -1)

        # OLD CODE COMMENTED OUT BELOW:
        # y_true_valid = y_true[valid_mask].cpu()
        # y_pred_valid = y_pred[valid_mask].cpu()

        y_true_valid = y_true[valid_mask].cpu()
        y_pred_valid = y_pred[valid_mask].cpu()
        y_score_valid = y_score[valid_mask].cpu()

        # OLD CODE COMMENTED OUT BELOW:
        # f1 = f1_score(y_true_valid, y_pred_valid, average='macro')
        # precision = precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
        # recall = recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)

        # NEW CODE =====================================================
        # Keep macro metrics for backward compatibility with the existing printouts.

        # მეტრიკების გამოთვლა (Macro ფოკუსირდება ყველა კლასზე თანაბრად)

        f1 = f1_score(y_true_valid, y_pred_valid, average='macro')
        precision = precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)

        # Minority-focused metrics for the illicit class (label = 1).
        f1_illicit = f1_score(y_true_valid, y_pred_valid, pos_label=1, average='binary', zero_division=0)
        precision_illicit = precision_score(y_true_valid, y_pred_valid, pos_label=1, average='binary', zero_division=0)
        recall_illicit = recall_score(y_true_valid, y_pred_valid, pos_label=1, average='binary', zero_division=0)

        pr_auc = average_precision_score(y_true_valid, y_score_valid)

        try:
            roc_auc = roc_auc_score(y_true_valid, y_score_valid)
        except ValueError:
            roc_auc = float("nan")
        # ==============================================================

        return {
            # Backward-compatible keys used by existing code
            "f1": f1,
            "precision": precision,
            "recall": recall,

            # NEW CODE =====================================================
            # Metrics required for the report / comparison against the baseline
            "f1_illicit": f1_illicit,
            "precision_illicit": precision_illicit,
            "recall_illicit": recall_illicit,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc
            # ==============================================================
        }