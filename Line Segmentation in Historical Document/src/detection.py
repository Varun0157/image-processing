import logging

import cv2
import numpy as np

from src.utils import show_image


# TODO: consider creating an iterable of ops and applying them. Cleaner?
# also, this is not final, just kinda works by accident.
# consider processing the top part and bottom part separately
def bounding_boxes(img: np.ndarray) -> np.ndarray:
    logging.info("pre-processing image for further analysis ... ")

    L = np.iinfo(img.dtype).max + 1
    y_sep = 80

    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(out, "gray", False, cmap="gray")

    # bottom half
    h, w, _ = img.shape
    bottom_area = (h - y_sep) * w

    out[y_sep:, :] = cv2.GaussianBlur(out[y_sep:, :], (5, 5), 0)  # kernel size, sigma
    show_image(out, "blurred", False, cmap="gray")

    _, out[y_sep:, :] = cv2.threshold(
        out[y_sep:, :], 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    show_image(out, "thresholded", False, cmap="gray")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out[y_sep:, :] = cv2.morphologyEx(out[y_sep:, :], cv2.MORPH_OPEN, kernel)
    show_image(out, "morphed", False, cmap="gray")

    contours, _ = cv2.findContours(
        out[y_sep:, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        logging.info(f"bounding box: {x, y, w, h}")

        if not (50 <= h * w <= (bottom_area / 2)):
            continue

        cv2.rectangle(img[y_sep:, :], (x, y), (x + w, y + h), (0, 255, 0), 2)
    show_image(img, "bounding boxes", False)

    return out
