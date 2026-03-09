# src/evaluate.py
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # ამოვიღოთ ნამდვილი და ნაწინასწარმეტყველები მონაცემები
        y_true = data.y[mask]
        y_pred = pred[mask]
        
        # გაფილტვრა -1-ებისგან
        valid_mask = (y_true != -1)
        
        y_true_valid = y_true[valid_mask].cpu()
        y_pred_valid = y_pred[valid_mask].cpu()
        
        # მეტრიკების გამოთვლა (Macro ფოკუსირდება ყველა კლასზე თანაბრად)
        f1 = f1_score(y_true_valid, y_pred_valid, average='macro')
        precision = precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
        
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }