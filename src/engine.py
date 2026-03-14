# src/engine.py

def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)

    supervised_mask = data.train_mask & (data.y != -1)

    loss = criterion(
        out[supervised_mask],
        data.y[supervised_mask]
    )

    loss.backward()
    optimizer.step()
    return loss.item()