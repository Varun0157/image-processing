# "give me opencv code to mask image outside a contour" - Claude AI


import cv2
import numpy as np


def mask_outside_contour(image, contour):
    # Create a black mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw white filled contour on mask
    cv2.drawContours(mask, [contour], 0, 255, -1)

    # Create white background
    white_background = np.full(image.shape, 255, dtype=np.uint8)

    # Copy image only where mask is nonzero
    result = np.where(mask[:, :, None], image, white_background)

    return result


# Example usage:
# img = cv2.imread('image.jpg')
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# result = mask_outside_contour(img, contours[0])
