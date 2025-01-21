import os

import torch

from utils.loops import train, evaluate
from utils.mlp import MLP
from utils.data import get_dataloader


def train_model(
    model: MLP,
    res_dir: str,
    data_path: str,
    criterion: torch.nn.NLLLoss,
    optim: Any,  # type: ignore TODO: fix
    epochs: int = 10,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    dropout_rate: float = 0.2,
    lr: float = 1e-3,
    sent_len: int | None = None,
) -> None:
    train_loader = get_dataloader(os.path.join(data_path, "train.csv"))
    valid_loader = get_dataloader(os.path.join(data_path, "valid.csv"))
    optim = optim(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_path = os.path.join(res_dir, "model.pth")

    for _ in range(epochs):
        _ = train(model, train_loader, optim, criterion, device)
        val_loss = evaluate(model, valid_loader, criterion, device)

        if val_loss > best_val_loss:
            continue

        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
