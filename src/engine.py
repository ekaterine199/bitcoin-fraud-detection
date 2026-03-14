# src/engine.py

def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)

    # OLD CODE COMMENTED OUT BELOW:
    # valid_mask = (data.y[data.train_mask] != -1)

    # NEW CODE =====================================================
    # Build a single supervised mask:
    #   - must be in the training split
    #   - must have a known label (not -1)
    supervised_mask = data.train_mask & (data.y != -1)
    # ==============================================================

    # OLD CODE COMMENTED OUT BELOW:
    # loss = criterion(
    #     out[data.train_mask][valid_mask], 
    #     data.y[data.train_mask][valid_mask]
    # )

    # NEW CODE =====================================================
    # Compute loss only on labeled training nodes.
    loss = criterion(
        out[supervised_mask],
        data.y[supervised_mask]
    )
    # ==============================================================

    loss.backward()
    optimizer.step()
    return loss.item()