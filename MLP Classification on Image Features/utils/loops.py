import torch
from torch.utils.data.dataloader import DataLoader

from utils.mlp import MLP


def train(
    model: MLP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,  # type: ignore
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    num_items: int = len(train_loader.dataset)  # type: ignore
    assert num_items > 0, "[train] training data must be present"
    model.train()

    total_loss = 0
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        output = model(image)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_items


def evaluate(
    model: MLP,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    num_items: int = len(test_loader.dataset)  # type: ignore
    assert num_items > 0, "[evaluate] testing data must be present"

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for context, target, _ in test_loader:
            context, target = context.to(device), target.to(device)

            output = model(context)
            loss = criterion(output, target)
            # NOTE: total loss of the batch, that's why we should use reduction="sum"

            total_loss += loss.item()

    return total_loss / num_items
