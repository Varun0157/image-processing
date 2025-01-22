import logging
from typing import Tuple, List

import cv2
import numpy as np

from src.utils import show_image


def text_segmentation(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    bbs_img = image.copy()
    ply_img = image.copy()

    logging.info("pre-processing image for further analysis ... ")

    L = np.iinfo(bbs_img.dtype).max + 1
    y_sep = 80

    final_text_contours: List[Tuple[int, int, int, int]] = []

    out = cv2.cvtColor(bbs_img, cv2.COLOR_BGR2GRAY)
    show_image(out, "gray", False, cmap="gray")

    # bottom half
    h, w, _ = bbs_img.shape
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

    main_doc_text_contours, _ = cv2.findContours(
        out[y_sep:, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in main_doc_text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        logging.info(f"bounding box: {x, y, w, h}")

        if not (50 <= h * w <= (bottom_area / 2)):
            continue

        cv2.rectangle(bbs_img[y_sep:, :], (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(ply_img[y_sep:, :], [contour], -1, (0, 255, 0), 1)

        final_text_contours.append((x, y + y_sep, w, h))
    show_image(bbs_img, "bounding boxes", False)

    # top half
    top_area = y_sep * w
    top_processed = out.copy()

    # mask the part outside the seal so we can make it white on binarisation
    _, out[:y_sep, :] = cv2.threshold(
        out[:y_sep, :], 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    show_image(out, "thresholded", False, cmap="gray")

    out[:y_sep, :] = cv2.morphologyEx(out[:y_sep, :], cv2.MORPH_OPEN, kernel)
    show_image(out, "morphed", False, cmap="gray")

    outside_seal_contours, _ = cv2.findContours(
        out[:y_sep, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    out[:y_sep, :] = cv2.bitwise_not(top_processed[:y_sep, :])
    show_image(out, "inverted", False, cmap="gray")

    for contour in outside_seal_contours:
        cv2.drawContours(out[:y_sep, :], [contour], -1, (255, 255, 255), -1)
    show_image(out, "filled", False, cmap="gray")

    # remove noise within seal
    out[:y_sep, :] = cv2.GaussianBlur(out[:y_sep, :], (5, 5), 0)
    show_image(out, "blurred", False, cmap="gray")

    out[:y_sep, :] = cv2.adaptiveThreshold(
        out[:y_sep, :], L, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    show_image(out, "thresholded", False, cmap="gray")

    # mask seal circle itself
    contours, _ = cv2.findContours(
        out[:y_sep, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        (x, y), rad = cv2.minEnclosingCircle(contour)
        x, y, rad = int(x), int(y), int(rad)

        # trying to mask away the seal circle
        if not (40 < y < 50) or rad < 15:
            continue

        mask = np.zeros(out[:y_sep, :].shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

        white_background = np.full(out[:y_sep, :].shape, L - 1, dtype=np.uint8)

        out[:y_sep, :] = np.where(mask[:, :], out[:y_sep, :], white_background)

    show_image(out, "without circles pls", False, cmap="gray")

    # finally, text segmentation
    seal_text_contours, _ = cv2.findContours(
        out[:y_sep, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in seal_text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        logging.info(f"bounding box: {x, y, w, h}")

        if not (10 <= h * w <= top_area / 2):
            continue

        cv2.rectangle(bbs_img[:y_sep, :], (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(ply_img[:y_sep, :], [contour], -1, (0, 255, 0), 1)

        final_text_contours.append((x, y, w, h))
    show_image(bbs_img, "bounding boxes", False)
    show_image(ply_img, "polygons", False)

    return bbs_img, ply_img, final_text_contours


def merge_adjacent_rects(
    img: np.ndarray,
    contours: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    def get_top_left_y(a: Tuple[int, int, int, int]) -> int:
        return a[1]

    contours.sort(key=get_top_left_y)
    final_bounds: List[Tuple[int, int, int, int]] = []
    assert len(contours) > 0, "no contours found"

    cur_x, cur_y, cur_w, cur_h = contours[0]

    for contour in contours[1:]:
        x, y, w, h = contour

        cur_mid = cur_y + cur_h / 2
        if y > cur_mid:
            final_bounds.append((cur_x, cur_y, cur_w, cur_h))
            cur_x, cur_y, cur_w, cur_h = x, y, w, h
        else:
            min_x = min(cur_x, x)
            min_y = min(cur_y, y)
            max_x = max(cur_x + cur_w, x + w)
            max_y = max(cur_y + cur_h, y + h)

            cur_x, cur_y = min_x, min_y
            cur_h, cur_w = max_y - min_y, max_x - min_x
    # NOTE: duplicates don't really matter here, so not thinking about it too mcuh
    final_bounds.append((cur_x, cur_y, cur_w, cur_h))

    for x, y, w, h in final_bounds:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img
