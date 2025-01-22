import logging

import cv2
import numpy as np

from src.utils import show_image


# TODO: consider creating an iterable of ops and applying them. Cleaner?
# also, this is not final, just kinda works by accident.
# consider processing the top part and bottom part separately
def polygon_bound(img: np.ndarray) -> np.ndarray:
    logging.info("pre-processing image for further analysis ... ")

    L = np.iinfo(img.dtype).max + 1

    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(out, "gray", False, cmap="gray")

    out = cv2.GaussianBlur(out, (5, 5), 0)  # kernel size, sigma
    show_image(out, "blurred", False, cmap="gray")

    _, out = cv2.threshold(out, 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image(out, "thresholded", False, cmap="gray")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    show_image(out, "morphed", False, cmap="gray")

    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show_image(img, "bounding boxes", False)

    return out
