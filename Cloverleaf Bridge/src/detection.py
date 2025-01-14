import logging

import cv2
import numpy as np

from src.utils import show_image


# TODO: consider creating an iterable of ops and applying them. Cleaner?
def preprocess_image(img: np.ndarray) -> np.ndarray:
    logging.info("pre-processing image for further analysis ... ")

    L = np.iinfo(img.dtype).max + 1

    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(out, "gray", False, cmap="gray")

    out = cv2.GaussianBlur(out, (5, 5), 0)  # kernel size, sigma
    show_image(out, "blurred", False, cmap="gray")

    _, out = cv2.threshold(out, 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image(out, "thresholded", False, cmap="gray")

    out = cv2.equalizeHist(out)
    show_image(out, "histogram equalised", False, cmap="gray")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    show_image(out, "morphed", False, cmap="gray")

    out = cv2.Canny(out, 30, 150)  # threshold1, threshold2
    show_image(out, "edge detected", False, cmap="gray")

    return out


# TODO: create a mark_radii option and use that to distinguish between 2.2.1 and 2.2.2
def mark_circles(img: np.ndarray) -> np.ndarray:
    logging.info("finding circles in the image ... ")
    processed = preprocess_image(img)

    # NOTE: issue faced - too many circles -> asked Claude for recommendations
    # show some of the other params messed with as well. Took a while to reach this result.
    # also mention that asking for how to find circles is what led you to HoughCircles in the first place.
    # show the code in the docs as a reference.
    circles = cv2.HoughCircles(
        processed,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=50,
        minRadius=150,
        maxRadius=300,
    )

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
