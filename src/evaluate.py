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

        # OLD CODE COMMENTED OUT BELOW:
        # y_true = data.y[mask]
        # y_pred = pred[mask]
        # y_score = torch.softmax(out[mask], dim=1)[:, 1]
        # valid_mask = (y_true != -1)
        # y_true_valid = y_true[valid_mask].cpu()
        # y_pred_valid = y_pred[valid_mask].cpu()
        # y_score_valid = y_score[valid_mask].cpu()

        # NEW CODE =====================================================
        valid_mask = mask & (data.y != -1)
        y_true_valid = data.y[valid_mask].cpu()
        y_pred_valid = pred[valid_mask].cpu()
        probs_valid = torch.softmax(out[valid_mask], dim=1).cpu()
        y_score_valid = probs_valid[:, 1]
        # =============================================================

        f1 = f1_score(y_true_valid, y_pred_valid, average='macro')
        precision = precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)

        # OLD CODE COMMENTED OUT BELOW:
        # f1_illicit = f1_score(y_true_valid, y_pred_valid, pos_label=1, average='binary', zero_division=0)
        # precision_illicit = precision_score(y_true_valid, y_pred_valid, pos_label=1, average='binary', zero_division=0)
        # recall_illicit = recall_score(y_true_valid, y_pred_valid, pos_label=1, average='binary', zero_division=0)
        # pr_auc = average_precision_score(y_true_valid, y_score_valid)

        # NEW CODE =====================================================
        # One-vs-rest illicit metrics still work whether we have 2 classes or 3.
        y_true_illicit = (y_true_valid == 1).int()
        y_pred_illicit = (y_pred_valid == 1).int()

        f1_illicit = f1_score(y_true_illicit, y_pred_illicit, zero_division=0)
        precision_illicit = precision_score(y_true_illicit, y_pred_illicit, zero_division=0)
        recall_illicit = recall_score(y_true_illicit, y_pred_illicit, zero_division=0)
        pr_auc = average_precision_score(y_true_illicit, y_score_valid)

        if torch.any(y_true_valid == 2):
            y_true_unknown = (y_true_valid == 2).int()
            y_pred_unknown = (y_pred_valid == 2).int()
            f1_unknown = f1_score(y_true_unknown, y_pred_unknown, zero_division=0)
        else:
            f1_unknown = float("nan")
        # =============================================================

        try:
            roc_auc = roc_auc_score(y_true_illicit, y_score_valid)
        except ValueError:
            roc_auc = float("nan")

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,

            "f1_illicit": f1_illicit,
            "precision_illicit": precision_illicit,
            "recall_illicit": recall_illicit,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,

            # NEW CODE =====================================================
            "macro_f1_all": f1,
            "f1_unknown": f1_unknown
            # =============================================================
        }