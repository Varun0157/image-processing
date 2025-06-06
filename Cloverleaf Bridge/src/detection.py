import logging
import math
from typing import Optional

import cv2
from cv2.typing import MatLike
import numpy as np

from src.utils import show_image


def preprocess_image(img: np.ndarray) -> np.ndarray:
    logging.info("pre-processing image for further analysis ... ")

    L = np.iinfo(img.dtype).max + 1

    out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    show_image(out, "processing - gray", True, cmap="gray")

    out = cv2.GaussianBlur(out, (5, 5), 0)  # kernel size, sigma
    show_image(out, "processing - blurred", True, cmap="gray")

    _, out = cv2.threshold(out, 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image(out, "processing - thresholded", True, cmap="gray")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    show_image(out, "processing - morphed - close", True, cmap="gray")

    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    show_image(out, "processing - morphed - open", True, cmap="gray")

    out = cv2.dilate(out, kernel, iterations=2)
    show_image(out, "processing - dilated", True, cmap="gray")

    return out


def detect_cloverleaves(out: np.ndarray) -> np.ndarray:
    def is_cloverleaf(contour: MatLike) -> bool:
        hull = cv2.convexHull(contour)

        perim = cv2.arcLength(hull, True)
        area = cv2.contourArea(hull)
        rad = perim / (2 + (3 / 2) * math.pi)

        # prevention of false positives
        if abs((area - cv2.contourArea(contour)) / area) > 0.2:
            return False

        # (x, y), enc_rad = cv2.minEnclosingCircle(contour)
        # rad = 2 * enc_rad / (1 + math.sqrt(2))
        # # NOTE: using the radius extracted from the min enclosing circle as above led
        # # to more accurate estimates of radius for the correct structures, but the incorrect ones too.
        # # using the perimeter led to the wrong ones being far more disparate.
        # is_cl_test = 200 <= enc_rad <= 250
        # # if is_cl_test and False:
        # #     cv2.circle(
        # #         out,
        # #         (int(x), int(y)),
        # #         radius=int(enc_rad),
        # #         color=(255, 255, 255),
        # #         thickness=2,
        # #     )

        tolerance = 0.10

        circle_area = (3 / 4) * math.pi * math.pow(rad, 2)
        square_area = math.pow(rad, 2)
        total_area = circle_area + square_area

        # # TODO: remove later
        # logging.info(
        #     f"{is_cl_test}\t: {(area - total_area) / total_area}, {area}, {total_area}"
        # )

        return abs((area - total_area) / total_area) < tolerance

    contours, _ = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        is_cl = is_cloverleaf(contour)

        colour = (0, 0, 0) if not is_cl else (255, 255, 255)
        cv2.drawContours(out, [contour], 0, colour, -1)

    show_image(out, "marking cloverleaves - filled", True, cmap="gray")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    show_image(out, "marking cloverleaves - morphed - close", True, cmap="gray")

    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    show_image(out, "marking cloverleaves - morphed - open", True, cmap="gray")

    return out


def find_circles(img: np.ndarray) -> Optional[np.ndarray]:
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=150,
        maxRadius=200,
    )

    return circles


def mark_circles(img: np.ndarray, circles: Optional[np.ndarray]) -> np.ndarray:
    if circles is None:
        logging.info("no circles found")
        return img

    annotated = img.copy()

    OUTER_COLOUR, CENTER_COLOUR = (0, 255, 0), (255, 0, 0)
    circles = np.round(circles).astype("int")
    for i in circles[0, :]:
        center, rad = (i[0], i[1]), i[2]
        cv2.circle(annotated, center, rad, OUTER_COLOUR, 2)
        cv2.circle(annotated, center, 2, CENTER_COLOUR, 3)

        cv2.putText(
            annotated,
            f"r={rad}",
            (center[0] + rad // 4, center[1] + rad // 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            OUTER_COLOUR,
            2,
        )

        center_text = f"({int(center[0])},{int(center[1])})"
        cv2.putText(
            annotated,
            center_text,
            (center[0] - rad // 4, center[1] - rad // 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            CENTER_COLOUR,
            2,
        )

        logging.info(f"\tcircle at {center_text} with radius {rad}")

    return annotated
