import time
import logging
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.utils import Colour


def get_histograms(
    img: np.ndarray,
) -> Tuple[Dict[Colour, np.ndarray], Dict[Colour, np.ndarray]]:
    logging.info("calculating histograms ... ")

    COLOURS = [Colour.BLUE, Colour.GREEN, Colour.RED]
    NUM_CHANNELS = img.shape[2]
    assert (
        NUM_CHANNELS == 3
    ), f"[calc_histogram] expected 3 channels (BGR), got {NUM_CHANNELS}"

    NUM_LEVELS = np.iinfo(img.dtype).max + 1

    custom_hist, custom_time = {}, 0
    opencv_hist, opencv_time = {}, 0
    for channel, colour in enumerate(COLOURS):
        start_time = time.time()
        custom_channel_hist = np.zeros(NUM_LEVELS)
        for intensity in img[:, :, channel].flatten():
            custom_channel_hist[intensity] += 1
        custom_hist[colour] = custom_channel_hist
        custom_time += time.time() - start_time

        start_time = time.time()
        opencv_channel_hist = cv2.calcHist(
            [img], [channel], None, [NUM_LEVELS], [0, NUM_LEVELS]
        ).flatten()
        opencv_hist[colour] = opencv_channel_hist
        opencv_time += time.time() - start_time

    logging.info("\ttime taken for custom histogram: %.4f", custom_time)
    logging.info("\ttime taken for opencv histogram: %.4f", opencv_time)

    logging.info("\tverifying correctness ... ")
    correct = all(np.array_equal(custom_hist[col], opencv_hist[col]) for col in COLOURS)
    logging.info("\t" + "correct!" if correct else "incorrect.")

    return custom_hist, opencv_hist


def visualise_histograms(
    custom: Dict[Colour, np.ndarray], opencv: Dict[Colour, np.ndarray]
) -> None:
    # TODO:
    # 1. add a mixed visualisation that shows the mix of all three in a single plot, for both
    # 2. add a binary version that convers to black and white and plots both - still maintain the columns, though.

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    for i, colour in enumerate([Colour.BLUE, Colour.GREEN, Colour.RED]):
        col_val = colour.value.lower()
        assert len(custom[colour]) == len(opencv[colour])
        NUM_LEVELS = len(custom[colour])

        axes[i, 0].bar(range(NUM_LEVELS), custom[colour], color=col_val, width=1)
        axes[i, 0].set_title(f"{col_val} - custom histogram")
        axes[i, 0].set_xlim(0, NUM_LEVELS - 1)

        axes[i, 1].bar(range(NUM_LEVELS), opencv[colour], color=col_val, width=1)
        axes[i, 1].set_title(f"{col_val} - opencv histogram")
        axes[i, 1].set_xlim(0, NUM_LEVELS - 1)
    fig.suptitle("Image Histograms")

    logging.info("showing histograms ... ")
    plt.show(block=True)
