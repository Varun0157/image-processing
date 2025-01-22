import os
from enum import Enum
import argparse

import torch
import wandb

from src.utils.loops import train_model, test_model
from src.utils.model import MLP
from src.utils.data import edge_detection, visualise, blurred_equalised, hog_features


class Transform(Enum):
    raw = "raw"
    edge_detect = "edge_detect"
    blur_equal = "blur_equal"
    hog_feat = "hog_feat"


def main(transform_type: Transform, batch_size: int, epochs: int, lr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    transform = None
    if transform_type == Transform.edge_detect:
        transform = edge_detection
    elif transform_type == Transform.blur_equal:
        transform = blurred_equalised
    elif transform_type == Transform.hog_feat:
        transform = hog_features

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
    parser = argparse.ArgumentParser(
        description="Train MLP classifier with different transformations"
    )

    parser.add_argument(
        "--transform",
        type=str,
        choices=[t.value for t in Transform],
        required=True,
        help="Type of transformation to apply to the data",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
        help="Batch size for training and testing",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )

    parser.add_argument("--lr", type=float, default=3.5e-5, help="Learning rate")

    args = parser.parse_args()

    transform_type = Transform(args.transform)

    main(
        transform_type=transform_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
