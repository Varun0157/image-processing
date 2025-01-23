import logging

from src.utils import load_document_image, show_image
from src.detection import (
    detect_lines_with_bboxes,
    preprocess_image,
    segment_lines_in_seal,
    detect_lines_with_polygons,
)
from src.histograms import get_histograms, visualise_histograms


def main() -> None:
    img = load_document_image()
    show_image(img, "initial image", save=False)

    custom_hist, opencv_hist = get_histograms(img)
    visualise_histograms(custom_hist, opencv_hist)

    y_sep = 80
    processed = preprocess_image(img, y_sep)
    show_image(processed, "final processed", save=True, cmap="gray")

    detect_lines_with_bboxes(processed, img, y_sep)
    segment_lines_in_seal(processed, img, y_sep)
    detect_lines_with_polygons(processed, img, y_sep)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )

    main()
