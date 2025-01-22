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

    return train_df, valid_df, test_df


if __name__ == "__main__":
    train_df, valid_df, test_df = prepare_data()
