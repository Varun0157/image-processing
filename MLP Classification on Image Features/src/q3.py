import os

import torch

from src.utils.model import train_model, test_model
from src.utils.common import MLP
from src.utils.data import edge_detection, visualise


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 2048

    visualise(os.path.join("data", "train.csv"), edge_detection)

    model = MLP(device=device)

    train_model(
        model=model,
        res_dir="res",
        data_path="data",
        transform=edge_detection,
        epochs=50,
        batch_size=batch_size,
        device=device,
    )

    test_model(
        model=model,
        res_dir="res",
        data_path="data",
        transform=edge_detection,
        batch_size=batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
