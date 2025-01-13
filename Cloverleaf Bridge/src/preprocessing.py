import cv2
import numpy as np

from src.utils import show_image


def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(gray, "gray image", cmap="gray")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # kernel size, sigma
    show_image(blurred, "blurred image", cmap="gray")

    high_contrast = cv2.equalizeHist(blurred)
    show_image(high_contrast, "high contrast image", cmap="gray")

    edge_detected = cv2.Canny(high_contrast, 30, 150)  # threshold1, threshold2
    show_image(edge_detected, "edge detected image", cmap="gray")

    out = edge_detected
    return out
