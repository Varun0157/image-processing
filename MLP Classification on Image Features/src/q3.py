import os
from typing import Optional, Callable

import torch
import numpy as np

from src.utils.loops import train_model, test_model
from src.utils.common import MLP
from src.utils.data import edge_detection, visualise, blurred_equalised

import wandb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 2048
    epochs: int = 100
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    lr: float = 3e-5

    run = wandb.init(
        project="basic mlp classification",
        config={
            "lr": lr,
            "epochs": epochs,
        },
        name=transform.__name__ if transform is not None else "raw",
    )

    visualise(os.path.join("data", "train.csv"), transform)

    model = MLP(device=device)

    train_model(
        model=model,
        res_dir="res",
        data_path="data",
        transform=transform,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        lr=lr,
    )

    test_model(
        model=model,
        res_dir="res",
        data_path="data",
        transform=transform,
        batch_size=batch_size,
        device=device,
    )

    run.finish()


if __name__ == "__main__":
    main()
