import os
from typing import Callable, Optional

import torch
from torch.optim.adam import Adam
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from src.utils.loops import train, evaluate
from src.utils.data import get_dataloader
from src.utils.common import MLP, get_model_path, get_num_classes


def train_model(
    model: MLP,
    res_dir: str,
    data_path: str,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    epochs: int = 10,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    lr: float = 6.5e-5,
) -> None:
    args = {
        "transform": transform,
        "batch_size": batch_size,
        "num_workers": 8,
        "shuffle": True,
    }
    train_loader = get_dataloader(os.path.join(data_path, "train.csv"), **args)
    valid_loader = get_dataloader(os.path.join(data_path, "valid.csv"), **args)

    optim = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    best_val_loss = float("inf")
    best_model_path = get_model_path(res_dir=res_dir, transform=transform)

    for epoch in range(epochs):
        print(f"\n*** EPOCH {epoch} ***")
        train_loss = train(model, train_loader, optim, criterion, device)
        val_loss, _ = evaluate(model, valid_loader, criterion, device)

        print(f"\ttrain loss: {train_loss:.4f}\tval loss: {val_loss:.4f}")
        wandb.log({"train loss": train_loss, "val loss": val_loss})

        if val_loss > best_val_loss:
            continue

        print("\tsaving model")

        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)


# NOTE: used Claude, copy prompt and res
def visualise_perf(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = get_num_classes(),
) -> None:
    conf_matrix = confusion_matrix(labels, preds)
    # TODO: check the average= param
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    accuracy = (preds == labels).mean()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": accuracy,
    }
    metrics_df = pd.DataFrame(
        {"Metric": metrics.keys(), "Value": [f"{v:.4f}" for v in metrics.values()]}
    )
    plt.axis("off")
    _ = plt.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        cellLoc="center",
        loc="center",
    )
    plt.title("Classification Metrics")

    plt.subplot(1, 2, 2)
    str_range = [str(label) for label in range(num_classes)]
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=str_range,
        yticklabels=str_range,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


def test_model(
    model: MLP,
    data_path: str,
    res_dir: str,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
) -> None:
    model.load_state_dict(
        torch.load(get_model_path(res_dir, transform), weights_only=True)
    )

    args = {
        "transform": transform,
        "batch_size": batch_size,
        "num_workers": 8,
    }
    test_loader = get_dataloader(os.path.join(data_path, "test.csv"), **args)

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    test_loss, (preds, labels) = evaluate(model, test_loader, criterion, device)
    print("\n*** TEST RESULTS ***")
    print("test loss: ", test_loss)

    visualise_perf(np.array(preds), np.array(labels))
