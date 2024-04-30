"""Preprocessing.py: helper functions for pre-processing."""
import numpy as np
import cv2 as cv

from typing import List
from sklearn.metrics.pairwise import pairwise_distances


def resize_image(image: np.ndarray, scale_percent: float = 0.8) -> np.ndarray:
    """
    Resize image for Hough Transform, to make it more efficient.

    Args:
        image (np.ndarray): image to be resized
        scale_percent (float): 1 = same size, 0.1 = 10% of original

    Returns:
        resized (np.ndarray): resized image
    """
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)

    resized = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    return resized


def filter_circles(hough_output: np.ndarray) -> np.ndarray:
    """
    If Hough Transform returns circles that overlap each other,
        filter them out and keep only the biggest circle to make
        sure that the coin is fully covered.

    Args:
        hough_output (np.ndarray): with shape (1, N, 3) or shape(N, 3)

    Returns:
        filtered_output (np.ndarray): with shape (K, 3)
    """
    # if shape is not (N, 3) make it so
    if len(hough_output.shape) != 2:
        hough_output = hough_output.squeeze(0)

    # make sure you have uint16 as dtype for cropping and plotting
    if hough_output.dtype != np.dtype('uint16'):
        hough_output = np.uint16(np.around(hough_output))

    # extract centers and radii
    centers, radii = hough_output[:, :2], hough_output[:, 2]

    distances = pairwise_distances(centers[:, :2])

    # get call the overlapping circles and clean bottom half
    is_inside = distances < radii
    is_inside[np.tril_indices(len(is_inside), 0)] = False

    # iterate over indices to find what to keep
    keep = np.full(len(hough_output), True)
    for i in range(len(hough_output)):

        if not keep[i]:
            continue

        # find all circles where i's center is inside and i is not the largest
        overlapping = is_inside[:, i]
        larger = radii[i] > radii[overlapping]
        if not all(larger):
            keep[i] = False

        # keep only the biggest circle
        keep[overlapping & (radii[i] >= radii)] = False

    return hough_output[keep]


def cut_coins(img: np.ndarray, coins_coords: np.ndarray, padding: int = 50) -> List[np.ndarray]:

    coins = []

    for (x, y, r) in coins_coords:

        r_pad = (r + padding)  # calculate radius with padding

        # get bounding box coordinates
        x_min, x_max = x - r_pad, x + r_pad
        y_min, y_max = y - r_pad, y + r_pad

        coin = img[y_min:y_max, x_min:x_max]
        coins.append(coin)

    return coins
