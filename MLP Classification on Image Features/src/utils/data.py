import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, csv_file: str, transform: None = None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.images = self.data.drop("label", axis=1).values
        self.labels = self.data["label"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


def get_dataloader(csv_file: str, transform: None = None, **kwargs) -> DataLoader:
    dataset = ImageDataset(csv_file, transform)
    return DataLoader(dataset, **kwargs)
