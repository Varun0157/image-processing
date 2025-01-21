import torch
from torch.optim.adam import Adam

from src.utils.mlp import MLP
from src.utils.model import train_model, test_model


def main():
    model = MLP(device=torch.device("cuda"))
    model = model.to(torch.device("cuda"))
    train_model(
        model,
        "res",
        "../data",
        torch.nn.CrossEntropyLoss(reduction="sum"),
        Adam,
        10,
        32,
        torch.device("cuda"),
        0.2,
        1e-3,
    )


if __name__ == "__main__":
    main()
