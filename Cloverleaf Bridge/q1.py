import logging

from src.utils import load_cloverfield_image, show_image
from src.histograms import calc_histogram, visualise_histograms
from src.preprocessing import preprocess_image


def main() -> None:
    img = load_cloverfield_image()
    show_image(img, "initial image")

    custom_hist, open_cv_hist = calc_histogram(img)
    visualise_histograms(custom_hist, open_cv_hist)

    _ = preprocess_image(img)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )

    main()
