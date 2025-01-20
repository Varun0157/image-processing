# create a function that prepares the data by randomly sampling 10% of the train images as the validation set and writes it to data/valid.csv and the new train to data/train.csv.
# In additino, I want to visualise the images. So, create a function that reshapes the images and stores them as images in data/vis/<<image>>. This can all be in a separate data.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path


def prepare_data(
    train_csv, test_csv, output_dir="data", valid_size=0.1, random_state=42
):
    """
    Split training data into train and validation sets and save to CSV files.

    Args:
        train_csv (str): Path to training CSV file
        test_csv (str): Path to test CSV file
        output_dir (str): Directory to save the processed data
        valid_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
    """
    # Create output directories
    data_dir = Path(output_dir)
    data_dir.mkdir(exist_ok=True)

    # Load data
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    # Split training data
    train_df, valid_df = train_test_split(
        train_data,
        test_size=valid_size,
        random_state=random_state,
        stratify=train_data["label"],
    )

    # Save split datasets
    train_df.to_csv(data_dir / "train.csv", index=False)
    valid_df.to_csv(data_dir / "valid.csv", index=False)
    test_data.to_csv(data_dir / "test.csv", index=False)

    print(f"Data split complete:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_data)}")

    return train_df, valid_df, test_data


def visualize_images(data_csv, output_dir="data/vis", num_samples=5, random_state=42):
    """
    Visualize random images from the dataset and save them.

    Args:
        data_csv (str): Path to CSV file containing image data
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of random images to visualize
        random_state (int): Random seed for reproducibility
    """
    # Create output directory
    vis_dir = Path(output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_csv(data_csv)

    # Set random seed
    np.random.seed(random_state)

    # Select random samples
    sample_indices = np.random.choice(len(data), num_samples, replace=False)

    for idx in sample_indices:
        # Get image data and label
        image_data = data.iloc[idx, 1:].values  # Skip label column
        label = data.iloc[idx, 0]

        # Reshape to 28x28 (assuming MNIST-like data)
        image = image_data.reshape(28, 28)

        # Create figure
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")

        # Save figure
        plt.savefig(
            vis_dir / f"sample_{idx}_label_{label}.png", bbox_inches="tight", dpi=100
        )
        plt.close()

    print(f"Saved {num_samples} sample images to {output_dir}")


if __name__ == "__main__":
    # Example usage
    train_df, valid_df, test_df = prepare_data("train.csv", "test.csv")
    visualize_images("data/train.csv", num_samples=10)
