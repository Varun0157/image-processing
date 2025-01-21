import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, csv_file: str, transform: None = None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.X = self.data.drop("label", axis=1).values
        self.y = self.data["label"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        img_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


def get_dataloader(csv_file: str, transform: None = None, **kwargs) -> DataLoader:
    dataset = ImageDataset(csv_file, transform)
    return DataLoader(dataset, **kwargs)
