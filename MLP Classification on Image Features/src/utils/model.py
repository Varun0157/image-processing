import os

import torch
from torch.optim.adam import Adam

from src.utils.loops import train, evaluate
from src.utils.mlp import MLP
from src.utils.data import get_dataloader


def train_model(
    model: MLP,
    res_dir: str,
    data_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    lr: float = 1e-3,
) -> None:
    args = {
        "transform": None,
        "batch_size": batch_size,
    }
    train_loader = get_dataloader(os.path.join(data_path, "train.csv"), **args)
    valid_loader = get_dataloader(os.path.join(data_path, "valid.csv"), **args)

    optim = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    best_val_loss = float("inf")
    best_model_path = os.path.join(res_dir, "model.pth")

    for epoch in range(epochs):
        print(f"\n*** EPOCH {epoch} ***")
        train_loss = train(model, train_loader, optim, criterion, device)
        val_loss, _ = evaluate(model, valid_loader, criterion, device)

        print(f"\ttrain loss: {train_loss:.4f}\tval loss: {val_loss:.4f}")

        if val_loss > best_val_loss:
            continue

        print("\tsaving model")

        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)


def test_model(
    model: MLP,
    data_path: str,
    res_dir: str,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
) -> None:
    model.load_state_dict(
        torch.load(os.path.join(res_dir, "model.pth"), weights_only=True)
    )

    args = {
        "transform": None,
        "batch_size": batch_size,
    }
    test_loader = get_dataloader(os.path.join(data_path, "test.csv"), **args)

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    test_loss, accuracy = evaluate(model, test_loader, criterion, device)
    print("\n*** TEST RESULTS ***")
    print("\ttest loss: {:.4f}\taccuracy: {:.4f}".format(test_loss, accuracy))
