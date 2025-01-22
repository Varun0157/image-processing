import logging

from src.utils import load_historical_doc, show_image
from src.detection import text_segmentation, merge_adjacent_rects
from src.histograms import get_histograms, visualise_histograms


def main() -> None:
    img = load_historical_doc()
    show_image(img, "initial image", save=False)

    custom_hist, opencv_hist = get_histograms(img)
    visualise_histograms(custom_hist, opencv_hist)

    bbs_img, ply_img, text_bounds = text_segmentation(img)
    show_image(bbs_img, "bounding boxes", True)
    show_image(ply_img, "polygons", True)

    line_wise_bbs_img = merge_adjacent_rects(img, text_bounds)
    show_image(line_wise_bbs_img, "line-wise bounding boxes", True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
    )

    main()
