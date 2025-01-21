import torch

from src.utils.model import train_model, test_model
from src.utils.common import MLP


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 512

    model = MLP(device=device)

    train_model(
        model=model,
        res_dir="res",
        data_path="data",
        epochs=25,
        batch_size=batch_size,
        device=device,
    )

    test_model(
        model=model,
        data_path="data",
        res_dir="res",
        batch_size=batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
