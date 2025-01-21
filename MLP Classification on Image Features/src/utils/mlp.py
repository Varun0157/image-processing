import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        num_classes=10,
        dropout=0.1,
        device=torch.device("cpu"),
    ):
        super(MLP, self).__init__()
        self.device = device

        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size, device=device),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_size = hidden_size
        layers.append(nn.Linear(current_size, num_classes, device=device))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        return self.layers(x)
