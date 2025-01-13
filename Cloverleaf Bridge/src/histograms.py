import logging
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.utils import Colour


def get_histograms(
    img: np.ndarray,
) -> Tuple[Dict[Colour, np.ndarray], Dict[Colour, np.ndarray]]:
    NUM_CHANNELS = img.shape[2]
    assert (
        NUM_CHANNELS == 3
    ), f"[calc_histogram] expected 3 channels (BGR), got {NUM_CHANNELS}"

    NUM_LEVELS = np.iinfo(img.dtype).max + 1

    custom_histograms = {}
    opencv_histograms = {}
    for i, colour in enumerate([Colour.BLUE, Colour.GREEN, Colour.RED]):
        channel = img[:, :, i]

        custom_channel_hist = np.zeros(NUM_LEVELS)
        for pixel in channel.flatten():
            custom_channel_hist[pixel] += 1
        custom_histograms[colour] = custom_channel_hist

        opencv_channel_hist = cv2.calcHist(
            [img], [i], None, [NUM_LEVELS], [0, NUM_LEVELS]
        ).flatten()

        opencv_histograms[colour] = opencv_channel_hist

    return custom_histograms, opencv_histograms


def visualise_histograms(
    custom: Dict[Colour, np.ndarray], opencv: Dict[Colour, np.ndarray]
) -> None:
    # TODO:
    # 1. add a mixed visualisation that shows the mix of all three in a single plot, for both
    # 2. add a binary version that convers to black and white and plots both - still maintain the columns, though.
    # 3. push the histograms relevant code to a new module entirely

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    for i, colour in enumerate([Colour.BLUE, Colour.GREEN, Colour.RED]):
        col_val = colour.value.lower()

        axes[i, 0].plot(custom[colour], color=col_val)
        axes[i, 0].set_title(f"{col_val} - custom histogram")
        axes[i, 0].set_xlim(0, 255)

        axes[i, 1].plot(opencv[colour], color=col_val)
        axes[i, 1].set_title(f"{col_val} - opencv histogram")
        axes[i, 1].set_xlim(0, 255)
    fig.suptitle("Image Histograms")

    logging.info("showing histograms ... ")
    plt.show(block=True)
