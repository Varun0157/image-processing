from typing import cast
from typing import Optional, Callable

from numpy.typing import ArrayLike
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.utils.common import get_num_classes, pixels_to_img


def edge_detection(pixels: np.ndarray) -> np.ndarray:
    image = pixels_to_img(pixels)
    out = cv2.Canny(image, 100, 200)

    return out.flatten()


def blurred_equalised(pixels: np.ndarray) -> np.ndarray:
    image = pixels_to_img(pixels)
    out = cv2.GaussianBlur(image, (3, 3), 0)
    out = cv2.equalizeHist(out)

    return out.flatten()


class ImageDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # NOTE: assuming intensities from 0 to 255
        self.images = self.data.drop("label", axis=1).values.astype(np.uint8)
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


def visualise(
    csv_file: str,
    transform: Optional[Callable[[np.ndarray], np.ndarray]],
):
    raw_dataset = ImageDataset(csv_file, None)
    transformed_dataset = ImageDataset(csv_file, transform)

    class_samples = {}
    labels = cast(ArrayLike, raw_dataset.labels)
    for class_id in range(get_num_classes()):
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) == 0:
            print(f"warning: no items of class id {class_id} found")

        class_samples[class_id] = np.random.choice(class_indices, 1)

    num_classes = len(class_samples)

    _, axes = plt.subplots(num_classes, 2, figsize=(6, 2 * num_classes))
    plt.suptitle("raw vs transformed images by class")

    axes[0, 0].set_title("raw")
    axes[0, 1].set_title("transformed")

    for row_idx, class_id in enumerate(class_samples.keys()):
        idx = class_samples[class_id]

        raw_img, _ = raw_dataset[idx]
        raw_img = pixels_to_img(raw_img.numpy())
        axes[row_idx, 0].imshow(raw_img, cmap="gray")
        axes[row_idx, 0].axis("off")

        # TODO: why isn't the label showing?
        axes[row_idx, 0].set_ylabel(f"{class_id}", rotation=0, ha="right", va="center")

        transformed_img, _ = transformed_dataset[idx]
        transformed_img = pixels_to_img(transformed_img.numpy())
        axes[row_idx, 1].imshow(transformed_img, cmap="gray")
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    plt.show()
