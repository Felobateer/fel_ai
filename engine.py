import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_prediction = model(X)
        
        loss = loss_fn(y_prediction, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_prediction_class = torch.argmax(torch.softmax(y_prediction, dim=1), dim=1)
        train_acc += (y_prediction_class == y).sum().item()/len(y_prediction)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

