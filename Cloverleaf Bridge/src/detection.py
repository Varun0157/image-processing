import logging
import math

import cv2
from cv2.typing import MatLike
import numpy as np

from src.utils import show_image


# TODO: consider creating an iterable of ops and applying them. Cleaner?
def detect_cloverleaves(img: np.ndarray) -> np.ndarray:
    logging.info("pre-processing image for further analysis ... ")

    L = np.iinfo(img.dtype).max + 1

    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(out, "gray", False, cmap="gray")

    out = cv2.GaussianBlur(out, (5, 5), 0)  # kernel size, sigma
    show_image(out, "blurred", False, cmap="gray")

    _, out = cv2.threshold(out, 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image(out, "thresholded", False, cmap="gray")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    show_image(out, "morphed - close", False, cmap="gray")

    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    show_image(out, "morphed - open", False, cmap="gray")

    out = cv2.equalizeHist(out)
    show_image(out, "histogram equalised", False, cmap="gray")

    # NOTE: idea was that when initially attempting to identify contours, a lot of the lines were "connected" and so multiple
    # structures that I did not want to be detected as one contour, were being detected as such.
    # Thus, I thought if I dilate, I make the internal circumference of the cloverleaves independent of the rest of the structure.
    out = cv2.dilate(out, kernel, iterations=2)
    show_image(out, "dilated", False, cmap="gray")

    def is_cloverleaf(contour: MatLike) -> bool:
        # NOTE: this idea consists of two parts:
        # 1. evaluating the convex hull of a contour (rather than the contour itself) massively decreases false negatives, because the
        # perimeter and area have to be close to that of a real cloverleaf. We extract radius from perimiter and compare area.
        # 2. there are rare cases where a cloverleaf like structure may have a large notch in them, making them create false positives
        # of cloverleaves where there may not be. Thus, we ensure low error between the area of the hull and the contour.

        hull = cv2.convexHull(contour)

        perim = cv2.arcLength(hull, True)
        area = cv2.contourArea(hull)
        rad = perim / (2 + (3 / 2) * math.pi)

        # prevention of false positives
        if abs((area - cv2.contourArea(contour)) / area) > 0.2:
            logging.info("seems like a potential false positive")
            return False

        (x, y), enc_rad = cv2.minEnclosingCircle(contour)
        rad = 2 * enc_rad / (1 + math.sqrt(2))
        # NOTE: using the radius extracted from the min enclosing circle as above led
        # to more accurate estimates of radius for the correct structures, but the incorrect ones too.
        # using the perimeter led to the wrong ones being far more disparate.
        is_cl_test = 200 <= enc_rad <= 250
        # if is_cl_test and False:
        #     cv2.circle(
        #         out,
        #         (int(x), int(y)),
        #         radius=int(enc_rad),
        #         color=(255, 255, 255),
        #         thickness=2,
        #     )

        tolerance = 0.10

        circle_area = (3 / 4) * math.pi * math.pow(rad, 2)
        square_area = math.pow(rad, 2)
        total_area = circle_area + square_area

        # TODO: remove later
        logging.info(
            f"{is_cl_test}\t: {(area - total_area) / total_area}, {area}, {total_area}"
        )

        return abs((area - total_area) / total_area) < tolerance

    contours, _ = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        is_cl = is_cloverleaf(contour)

        colour = (0, 0, 0) if not is_cl else (255, 255, 255)
        cv2.drawContours(out, [contour], 0, colour, -1)

    show_image(out, "contours filled", False, cmap="gray")

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    show_image(out, "morphed - close", False, cmap="gray")

    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    show_image(out, "morphed - open", False, cmap="gray")

    return out


def _find_circles(img: np.ndarray) -> np.ndarray | None:
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


def _mark_circles(img: np.ndarray, circles: np.ndarray | None) -> np.ndarray:
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


# TODO: create a mark_radii option and use that to distinguish between 2.2.1 and 2.2.2
def mark_circles(img: np.ndarray) -> np.ndarray:
    logging.info("finding circles in the raw image ... ")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = _find_circles(gray)
    annotated = _mark_circles(img, circles)
    show_image(annotated, "raw circles", save=True, cmap="gray")

    logging.info("processing images and finding circles in the image ... ")
    processed = detect_cloverleaves(img)

    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    annotated_contours = img.copy()
    cv2.drawContours(annotated_contours, contours, -1, (255, 0, 0), 1)
    show_image(annotated_contours, "contour borders", save=True)

    circles = _find_circles(processed)
    out = _mark_circles(img, circles)

    return out
