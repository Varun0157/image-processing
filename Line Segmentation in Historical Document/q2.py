import logging

from src.utils import load_historical_doc, show_image
from src.detection import bounding_boxes
from src.histograms import get_histograms, visualise_histograms


def main() -> None:
    img = load_historical_doc()
    show_image(img, "initial image", save=False)

    custom_hist, opencv_hist = get_histograms(img)
    visualise_histograms(custom_hist, opencv_hist)

    annotated = bounding_boxes(img)
    show_image(annotated, "processed", False, cmap="gray")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )

    main()
