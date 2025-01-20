from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.X = self.data.drop("label", axis=1).values
        self.y = self.data["label"].values

    def __len__(self):
        return len(self.data)

    # TODO: simplify
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.X[idx].astype(np.float32) / 255.0
        image = torch.FloatTensor(image)

        label = self.y[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
