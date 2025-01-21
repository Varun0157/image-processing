import os
from typing import Optional

import torch.nn as nn
import torch


def get_model_path(res_dir: str, transform_name: Optional[str] = None) -> str:
    model_components = ["model"]
    if transform_name is not None:
        model_components.append(transform_name)
    model_components.append(".pth")
    model_name = ".".join(model_components)

    return os.path.join(res_dir, model_name)


def get_input_size() -> int:
    return 784


def get_num_classes() -> int:
    return 10


class MLP(nn.Module):
    def __init__(
        self,
        input_size=get_input_size(),
        hidden_sizes=[512, 256, 128, 256],
        num_classes=get_num_classes(),
        dropout=0.1,
        device=torch.device("cpu"),
    ):
        super(MLP, self).__init__()
        self.device = device

        self.layers = nn.ModuleList()
        current_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.extend(
                [
                    nn.Linear(current_size, hidden_size, device=device),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_size = hidden_size
        self.layers.append(nn.Linear(current_size, num_classes, device=device))

    def forward(self, x):
        out = x.to(self.device)
        for layer in self.layers:
            out = layer(out)

        return out
