import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(
    data_dir="raw",
    target_dir="data",
    valid_size: float = 0.1,
    random_state: int = 42,
):
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # split training data into train and valid
    train_df, valid_df = train_test_split(
        train_data,
        test_size=valid_size,
        random_state=random_state,
        stratify=train_data["label"],
    )
    test_df = test_data

    assert type(train_df) is pd.DataFrame
    assert type(test_df) is pd.DataFrame
    assert type(valid_df) is pd.DataFrame

    # TODO: does / use an OS specific path separator?
    targ_path = Path(target_dir)
    targ_path.mkdir(exist_ok=True)

    train_df.to_csv(targ_path / "train.csv", index=False)
    valid_df.to_csv(targ_path / "valid.csv", index=False)
    test_df.to_csv(targ_path / "test.csv", index=False)

    print("Data split complete:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_data)}")
    # print(f"train labels: {train_df.label.unique()}")
    # print(f"valid labels: {valid_df.label.unique()}")

    return train_df, valid_df, test_data


def visualize_images(
    data_csv, output_dir=os.path.join("data", "vis"), num_samples=5, random_state=42
):
    vis_dir = Path(output_dir)
    # TODO: parents=True?
    vis_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_csv)

    np.random.seed(random_state)

    sample_indices = np.random.choice(len(data), num_samples, replace=False)

    for idx in sample_indices:
        image_data = data.iloc[idx, 1:].values  # Skip label column
        label = data.iloc[idx, 0]

        image = image_data.reshape(28, 28)

        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")

        plt.savefig(
            vis_dir / f"sample_{idx}_label_{label}.png", bbox_inches="tight", dpi=100
        )
        plt.close()

    print(f"saved {num_samples} sample images to {output_dir}")


if __name__ == "__main__":
    train_df, valid_df, test_df = prepare_data()
    visualize_images("data/train.csv", num_samples=10)
