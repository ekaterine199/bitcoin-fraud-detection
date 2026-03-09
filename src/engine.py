def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    
    valid_mask = (data.y[data.train_mask] != -1)
    
    loss = criterion(
        out[data.train_mask][valid_mask], 
        data.y[data.train_mask][valid_mask]
    )
    
    loss.backward()
    optimizer.step()
    return loss.item()