import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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

    return train_df, valid_df, test_df


def store_images(data_dir: str = "data") -> None:
    from PIL import Image

    src_path = os.path.join("data", "train.csv")

    df = pd.read_csv(src_path)
    images = df.drop("label", axis=1).values.astype(np.uint8)
    labels = df["label"].values

    images_dir = Path(data_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(images)):
        label = labels[index]
        pixels = images[index]

        image = (pixels.reshape(28, 28)).astype(np.uint8)

        label_dir = images_dir / str(label)
        label_dir.mkdir(exist_ok=True)

        img = Image.fromarray(image)
        img.save(label_dir / f"{index}.png")

    print(f"images from {src_path} stored in {images_dir}")


if __name__ == "__main__":
    train_df, valid_df, test_df = prepare_data()
    store_images()
