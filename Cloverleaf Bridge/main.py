import logging
import math
from typing import Optional, Sequence, Tuple

import cv2
from cv2.typing import MatLike
import numpy as np

from src.utils import load_cloverfield_image, show_image
from src.histograms import get_histograms, visualise_histograms
from src.detection import (
    find_circles,
    mark_circles,
    preprocess_image,
    detect_cloverleaves,
)


def calculate_radii(
    image: np.ndarray, processed: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    img = image.copy()

    logging.info("marking circles in original image ... ")
    circles = find_circles(processed)
    out = mark_circles(img, circles)
    show_image(out, "radii", save=True)

    return out, circles


def calculate_area(
    circles: Optional[np.ndarray], cloverleaves: Sequence[MatLike]
) -> None:
    logging.info("areas reported by hough circles (pixels squared): ")
    if circles is None:
        logging.info("\tno circles found")
    else:
        circles = np.round(circles).astype("int")
        for circle in circles[0, :]:
            rad = circle[2]
            circular_area = (3 / 4) * math.pi * math.pow(rad, 2)
            square_area = math.pow(rad, 2)
            area = circular_area + square_area

            logging.info(f"\tarea: {area:.6f}")

    logging.info("areas reported by contours (pixels squared): ")
    for cloverleaf in cloverleaves:
        logging.info("\tarea of: ")
        logging.info(f"\t\tdetected contour: {cv2.contourArea(cloverleaf):.6f}")
        hull = cv2.convexHull(cloverleaf)
        logging.info(f"\t\tsurrounding hull: {cv2.contourArea(hull):.6f}")


def main() -> None:
    img = load_cloverfield_image()
    show_image(img, "initial image", save=True)

    # logging.info("just to check: calc cloverleaves on grayed image")
    # processed = detect_cloverleaves(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # show_image(processed, "filled cloverleaves on grayed", save=True, cmap="gray")

    custom_hist, open_cv_hist = get_histograms(img)
    visualise_histograms(custom_hist, open_cv_hist)

    logging.info("finding circles in the raw image ... ")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    circles = find_circles(gray)
    annotated = mark_circles(img, circles)
    show_image(annotated, "raw circles", save=True, cmap="gray")

    logging.info("processing image ... ")
    processed = preprocess_image(img)
    show_image(processed, "final processed", save=True, cmap="gray")
    logging.info("filling cloverleaves in processed image ... ")
    processed = detect_cloverleaves(processed)
    show_image(processed, "filled cloverleaves", save=True, cmap="gray")

    logging.info("marking cloverleaf borders in original image ... ")
    cloverleaves, _ = cv2.findContours(
        processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    annotated_contours = img.copy()
    cv2.drawContours(annotated_contours, cloverleaves, -1, (255, 0, 0), 3)
    show_image(annotated_contours, "contour borders", save=True)

    _, circles = calculate_radii(img, processed)

    calculate_area(circles, cloverleaves)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )

    main()
