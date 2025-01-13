import cv2
import numpy as np

from src.utils import show_image


# TODO: consider creating an iterable of ops and applying them. Cleaner.
def preprocess_image(img: np.ndarray) -> np.ndarray:
    L = np.iinfo(img.dtype).max + 1

    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(out, "gray", cmap="gray")

    out = cv2.GaussianBlur(out, (5, 5), 0)  # kernel size, sigma
    show_image(out, "blurred", cmap="gray")

    _, out = cv2.threshold(out, 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image(out, "thresholded", cmap="gray")

    out = cv2.equalizeHist(out)
    show_image(out, "histogram equalised", cmap="gray")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    show_image(out, "morphed", cmap="gray")

    out = cv2.Canny(out, 30, 150)  # threshold1, threshold2
    show_image(out, "edge detected", cmap="gray")

    return out
