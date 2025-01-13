import logging

from src.utils import load_cloverfield_image, show_image
from src.histograms import get_histograms, visualise_histograms
from src.detection import find_circles


def main() -> None:
    img = load_cloverfield_image()
    show_image(img, "initial image", save=False)

    custom_hist, open_cv_hist = get_histograms(img)
    visualise_histograms(custom_hist, open_cv_hist)

    annotated = find_circles(img)
    show_image(annotated, "with circles", save=True, cmap="gray")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )

    main()
