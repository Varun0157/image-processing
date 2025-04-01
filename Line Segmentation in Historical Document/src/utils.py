from enum import Enum
import logging
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_save_path() -> str:
    return "res"


class Colour(Enum):
    RED = "RED"
    BLUE = "BLUE"
    GREEN = "GREEN"


def load_image(path: str, log_info: bool = True) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if log_info:
        logging.info(f"loaded image from {path}")
        logging.info(f"\tshape: {img.shape}")
        logging.info(f"\tdtype: {img.dtype}")
        logging.info(f"\tmin: {img.min()}")
        logging.info(f"\tmax: {img.max()}")
        logging.info(f"\tmean: {img.mean()}")

    return img


def load_document_image() -> np.ndarray:
    IMAGE_PATH = os.path.join("data", "historical-doc.png")
    img = load_image(IMAGE_PATH, log_info=True)

    return img


def show_image(img: np.ndarray, title: str, save: bool, **im_kwargs) -> None:
    logging.info(f"showing image - title: {title} ... ")
    plt.imshow(img, **im_kwargs)
    plt.title(title)
    plt.show(block=True)

    if not save:
        return

    file_path = os.path.join(get_save_path(), f"{title}.png")
    plt.imsave(file_path, img, **im_kwargs)
    logging.info(f"saved image to {file_path}")
