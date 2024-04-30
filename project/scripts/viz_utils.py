"""viz_utils.py: visualization helpers for the project."""
import numpy as np
from matplotlib import pyplot as plt


def show_image(img: np.ndarray, is_rgb: bool = True):
    """
    Plot the image and take rgb order into account

    Args:
        img (np.ndarray): image to be plot
        is_rgb: True if channel order is red-green-blue, False otherwise

    Returns:
        plots the image
    """
    if not is_rgb:
        img = img[:, :, ::-1]  # bgr -> rgb

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
