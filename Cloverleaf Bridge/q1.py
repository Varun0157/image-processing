import logging

from src.utils import load_cloverfield_image, show_image
from src.histograms import get_histograms, visualise_histograms
from src.preprocessing import preprocess_image


def main() -> None:
    img = load_cloverfield_image()
    show_image(img, "initial image", save=False)

    custom_hist, open_cv_hist = get_histograms(img)
    visualise_histograms(custom_hist, open_cv_hist)

    preprocessed = preprocess_image(img)
    show_image(preprocessed, "pre-processed", save=True, cmap="gray")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )

    main()
