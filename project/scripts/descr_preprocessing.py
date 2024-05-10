import cv2

import numpy as np
import scripts.descriptors as d

from skimage import draw
from skimage.color import *

def extract_circular_region(img, radius=100):
	"""
	Extract part of an image as a circular region centred at image center
	"""

	height, width = img.shape[0:-1]
	xcenter = width // 2
	ycenter = height // 2

	# Draw circle
	canvas = np.zeros((height, width))
	cv2.circle(canvas, (xcenter, ycenter), radius, (255,255,255), -1)
	
	# Create a copy of the input and mask input:
	imageCopy = img.copy()
	imageCopy[canvas == 0] = (0, 0, 0)

	# Crop the roi:
	x = xcenter - radius
	y = ycenter - radius
	h = 2 * radius
	w = 2 * radius

	croppedImg = imageCopy[y:y + h, x:x + w]

	return croppedImg

def get_descriptors(img, descriptor_list):
	"""
	For an image, compute all its descriptors given in parameter.

	Returns a dict
	"""

	descriptor_values = {}

	for descriptor in descriptor_list:
		descriptor_function = getattr(d, str(descriptor))
		descriptor_values[descriptor] = descriptor_function(img)

	return descriptor_values