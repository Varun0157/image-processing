import logging

from src.utils import load_historical_doc, show_image


def main() -> None:
    img = load_historical_doc()
    show_image(img, "initial image", save=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )

    main()
