from typing import Tuple, List

from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

from src.utils.common import MLP


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
    for image, label in tqdm(train_loader, "training .... "):
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
) -> Tuple[float, Tuple[List[float], List[float]]]:
    num_items: int = len(test_loader.dataset)  # type: ignore
    assert num_items > 0, "[evaluate] testing data must be present"

    model.eval()

    total_loss = 0
    preds, labels = [], []

    with torch.no_grad():
        for image, label in tqdm(test_loader, "evaluating .. "):
            image, label = image.to(device), label.to(device)

            output = model(image)

            loss = criterion(output, label)
            total_loss += loss.item()

            pred = output.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())

    avg_loss = total_loss / num_items

    return avg_loss, (preds, labels)
