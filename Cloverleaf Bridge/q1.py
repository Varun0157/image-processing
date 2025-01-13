import os
import logging
from typing import Dict, Tuple
from enum import Enum

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Colour(Enum):
    RED = "RED"
    BLUE = "BLUE"
    GREEN = "GREEN"


def load_image(path: str, log_info: bool = True) -> np.ndarray:
    img = cv2.imread(path)

    if log_info:
        logging.info(f"loaded image from {path}")
        logging.info(f"\tshape: {img.shape}")
        logging.info(f"\tdtype: {img.dtype}")
        logging.info(f"\tmin: {img.min()}")
        logging.info(f"\tmax: {img.max()}")
        logging.info(f"\tmean: {img.mean()}")

    return img


def calc_histogram(
    img: np.ndarray,
) -> Tuple[Dict[Colour, np.ndarray], Dict[Colour, np.ndarray]]:
    NUM_CHANNELS = img.shape[2]
    assert (
        NUM_CHANNELS == 3
    ), f"[calc_histogram] expected 3 channels (BGR), got {NUM_CHANNELS}"

    NUM_LEVELS = np.iinfo(img.dtype).max + 1

    custom_histograms = {}
    opencv_histograms = {}
    for i, colour in enumerate([Colour.BLUE, Colour.GREEN, Colour.RED]):
        channel = img[:, :, i]

        custom_channel_hist = np.zeros(NUM_LEVELS)
        for pixel in channel.flatten():
            custom_channel_hist[pixel] += 1
        custom_histograms[colour] = custom_channel_hist

        opencv_channel_hist = cv2.calcHist(
            [img], [i], None, [NUM_LEVELS], [0, NUM_LEVELS]
        ).flatten()

        opencv_histograms[colour] = opencv_channel_hist

    return custom_histograms, opencv_histograms


def visualise_histograms(
    custom: Dict[Colour, np.ndarray], opencv: Dict[Colour, np.ndarray]
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    for i, colour in enumerate([Colour.BLUE, Colour.GREEN, Colour.RED]):
        col_val = colour.value.lower()

        axes[i, 0].plot(custom[colour], color=col_val)
        axes[i, 0].set_title(f"{col_val} - custom histogram")
        axes[i, 0].set_xlim(0, 255)

        axes[i, 1].plot(opencv[colour], color=col_val)
        axes[i, 1].set_title(f"{col_val} - opencv histogram")
        axes[i, 1].set_xlim(0, 255)
    fig.tight_layout()

    logging.info("showing histograms ... ")
    plt.show()


def show_image(img: np.ndarray) -> None:
    # TODO: create a title param and add it to the image plot
    plt.imshow(img)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )
    IMAGE_PATH = os.path.join("data", "cloverleaf_interchange.png")

    # TODO: make the images show in a blocking manner rather than all at once
    img = load_image(IMAGE_PATH)
    show_image(img)

    custom_hist, open_cv_hist = calc_histogram(img)
    visualise_histograms(custom_hist, open_cv_hist)
