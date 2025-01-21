import torch

from src.utils.mlp import MLP
from src.utils.model import train_model, test_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(device=device)
    train_model(
        model,
        "res",
        "data",
        10,
        32,
        device,
        1e-3,
    )

    test_model(model, "data", "res", 32, device)


if __name__ == "__main__":
    main()
