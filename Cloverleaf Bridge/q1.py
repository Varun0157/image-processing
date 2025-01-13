import os
import logging

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)

    # TODO: iterate through these attributes instead of repetitive naming
    logging.info(f"loaded image from {path}")
    logging.info(f"\tshape: {img.shape}")
    logging.info(f"\tdtype: {img.dtype}")
    logging.info(f"\tmin: {img.min()}")
    logging.info(f"\tmax: {img.max()}")
    logging.info(f"\tmean: {img.mean()}")

    return img


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    img = load_image(os.path.join("data", "cloverleaf_interchange.png"))
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
