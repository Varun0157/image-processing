import os
import logging
from typing import Dict

import cv2
import numpy as np


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


def calc_histogram(img: np.ndarray) -> Dict[str, np.ndarray]:
    NUM_CHANNELS = img.shape[2]
    assert (
        NUM_CHANNELS == 3
    ), f"[calc_histogram] expected 3 channels (BGR), got {NUM_CHANNELS}"

    NUM_LEVELS = np.iinfo(img.dtype).max + 1

    histograms = {}
    for i, colour in enumerate(["blue", "green", "red"]):
        channel = img[:, :, i]

        channel_hist = np.zeros(NUM_LEVELS)
        for pixel in channel.flatten():
            channel_hist[pixel] += 1
        histograms[colour] = channel_hist

    return histograms


def show_image(img: np.ndarray, title: str) -> None:
    cv2.imshow(title, img)
    cv2.waitKey(0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )
    IMAGE_PATH = os.path.join("data", "cloverleaf_interchange.png")

    img = load_image(IMAGE_PATH)
    show_image(img, "initial image")

    cv2.destroyAllWindows()
