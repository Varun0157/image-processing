from enum import Enum
import logging
import os

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
        # TODO: clarify what else is meant by 'distribution'
        logging.info(f"loaded image from {path}")
        logging.info(f"\tshape: {img.shape}")
        logging.info(f"\tdtype: {img.dtype}")
        logging.info(f"\tmin: {img.min()}")
        logging.info(f"\tmax: {img.max()}")
        logging.info(f"\tmean: {img.mean()}")

    return img


def load_cloverfield_image() -> np.ndarray:
    IMAGE_PATH = os.path.join("data", "cloverleaf_interchange.png")
    img = load_image(IMAGE_PATH, log_info=True)

    return img


def show_image(img: np.ndarray, title: str, **imshow_kwargs) -> None:
    # TODO: make it an option to save image under title
    logging.info(f"showing image - title: {title} ... ")
    plt.imshow(img, **imshow_kwargs)
    plt.title(title)
    plt.show(block=True)
