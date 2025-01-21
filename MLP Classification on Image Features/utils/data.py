from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, csv_file: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.X = self.data.drop("label", axis=1).values
        self.y = self.data["label"].values

    def __len__(self):
        return len(self.data)

    # TODO: simplify
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(csv_file: str, **kwargs) -> DataLoader:
    dataset = ImageDataset(csv_file)
    return DataLoader(dataset, **kwargs)
