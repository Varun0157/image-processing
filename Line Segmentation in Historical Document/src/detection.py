import logging
from typing import Tuple, List

import cv2
import numpy as np

from src.utils import show_image


def _mark_main_doc_contours(
    y_sep: int, processed: np.ndarray, img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    bbs_img = img.copy()
    ply_img = img.copy()
    final_text_contours: List[Tuple[int, int, int, int]] = []

    h, w, _ = img.shape
    bottom_area = (h - y_sep) * w

    main_doc_text_contours, _ = cv2.findContours(
        processed[y_sep:, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in main_doc_text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        logging.info(f"bounding box: {x, y, w, h}")

        if not (50 <= h * w <= (bottom_area / 2)):
            continue

        cv2.rectangle(bbs_img[y_sep:, :], (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(ply_img[y_sep:, :], [contour], -1, (0, 255, 0), 1)

        final_text_contours.append((x, y + y_sep, w, h))

    return bbs_img, ply_img, final_text_contours


def _mark_seal_text_contours(
    y_sep: int, processed: np.ndarray, img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    bbs_img = img.copy()
    ply_img = img.copy()
    final_text_contours: List[Tuple[int, int, int, int]] = []

    h, w, _ = img.shape
    top_area = y_sep * w

    seal_text_contours, _ = cv2.findContours(
        processed[:y_sep, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in seal_text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        logging.info(f"bounding box: {x, y, w, h}")

        if not (10 <= h * w <= top_area / 2):
            continue

        cv2.rectangle(bbs_img[:y_sep, :], (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(ply_img[:y_sep, :], [contour], -1, (0, 255, 0), 1)

        final_text_contours.append((x, y, w, h))

    return bbs_img, ply_img, final_text_contours


def _merge_adjacent_rects(
    image: np.ndarray,
    rects: List[Tuple[int, int, int, int]],
    name: str,
) -> np.ndarray:
    img = image.copy()

    def get_top_left_y(a: Tuple[int, int, int, int]) -> int:
        return a[1]

    rects.sort(key=get_top_left_y)
    final_bounds: List[Tuple[int, int, int, int]] = []
    assert len(rects) > 0, "no contours found"

    cur_x, cur_y, cur_w, cur_h = rects[0]

    for contour in rects[1:]:
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
    # but, a duplicate push is possible here. A simple fix would be to make it a set,
    # or explicitly check for presence.
    final_bounds.append((cur_x, cur_y, cur_w, cur_h))

    for i, (x, y, w, h) in enumerate(final_bounds):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        raw_img = image.copy()
        cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show_image(raw_img, f"{name}_{i+1}", save=True)

    return img


# NOTE: the following three functions are present to satisfy requirements in the document
# They make repeated calls to the same function, which is un-necessary and will be removed later.
def detect_lines_with_bboxes(
    processed: np.ndarray, img: np.ndarray, y_sep: int = 80
) -> None:
    bbs_img, _, rects = _mark_main_doc_contours(y_sep, processed, img)
    show_image(bbs_img, "text-wise bounding boxes - document", save=True)

    line_bbs_img = _merge_adjacent_rects(img, rects, "line")
    show_image(line_bbs_img, "line-wise bounding boxes - document", save=True)


def segment_lines_in_seal(
    processed: np.ndarray, img: np.ndarray, y_sep: int = 80
) -> None:
    bbs_img, _, rects = _mark_seal_text_contours(y_sep, processed, img)
    show_image(bbs_img, "text-wise bounding boxes - seal", save=True)

    line_bbs_img = _merge_adjacent_rects(img, rects, "circle_line")
    show_image(line_bbs_img, "line-wise bounding boxes - seal", save=True)


def detect_lines_with_polygons(
    processed: np.ndarray, img: np.ndarray, y_sep: int = 80
) -> None:
    _, ply_img, _ = _mark_main_doc_contours(y_sep, processed, img)
    _, ply_img, _ = _mark_seal_text_contours(y_sep, processed, ply_img)

    show_image(ply_img, "text-wise polygons - document", save=True)


def preprocess_image(image: np.ndarray, y_sep: int = 80) -> np.ndarray:
    logging.info("pre-processing image for further analysis ... ")

    L = np.iinfo(image.dtype).max + 1

    out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(out, "full - gray", False, cmap="gray")

    # bottom half
    out[y_sep:, :] = cv2.GaussianBlur(out[y_sep:, :], (5, 5), 0)  # kernel size, sigma
    show_image(out, "bottom - blurred", False, cmap="gray")

    _, out[y_sep:, :] = cv2.threshold(
        out[y_sep:, :], 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    show_image(out, "bottom - thresholded", False, cmap="gray")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out[y_sep:, :] = cv2.morphologyEx(out[y_sep:, :], cv2.MORPH_OPEN, kernel)
    show_image(out, "bottom - morphed", False, cmap="gray")

    # top half
    top_processed = out.copy()

    # mask the part outside the seal so we can make it white on binarisation
    _, out[:y_sep, :] = cv2.threshold(
        out[:y_sep, :], 0, L, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    show_image(out, "top - thresholded", False, cmap="gray")

    out[:y_sep, :] = cv2.morphologyEx(out[:y_sep, :], cv2.MORPH_OPEN, kernel)
    show_image(out, "top - morphed", False, cmap="gray")

    outside_seal_contours, _ = cv2.findContours(
        out[:y_sep, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    out[:y_sep, :] = cv2.bitwise_not(top_processed[:y_sep, :])
    show_image(out, "top - inverted", False, cmap="gray")

    for contour in outside_seal_contours:
        cv2.drawContours(out[:y_sep, :], [contour], -1, (255, 255, 255), -1)
    show_image(out, "top - outside seal filled", False, cmap="gray")

    # remove noise within seal
    out[:y_sep, :] = cv2.GaussianBlur(out[:y_sep, :], (5, 5), 0)
    show_image(out, "top - blurred", False, cmap="gray")

    out[:y_sep, :] = cv2.adaptiveThreshold(
        out[:y_sep, :], L, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    show_image(out, "top - thresholded (adaptive)", False, cmap="gray")

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

    show_image(out, "top - without seal circle", False, cmap="gray")

    return out
