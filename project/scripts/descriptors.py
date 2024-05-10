import numpy as np
from scripts.descr_preprocessing import extract_circular_region

def descr_area(radius):
	return np.pi * radius**2

def descr_mean(img):
	"""
	Returns means of channels of an image

	(C1_mu, C2_mu, C3_mu)
	"""
	c1 = np.copy(img[:,:,0])
	c2 = np.copy(img[:,:,1])
	c3 = np.copy(img[:,:,2])

	return c1.mean(), c2.mean(), c3.mean()

def descr_std(img):
	"""
	Returns std of channels of an image

	(C1_std, C2_std, C3_std)
	"""
	c1 = np.copy(img[:,:,0])
	c2 = np.copy(img[:,:,1])
	c3 = np.copy(img[:,:,2])

	return c1.std(), c2.std(), c3.std()

def descr_mean_center(img, radius):
	"""
	Extract part of the center to compute its color mean

	"""
	img_cropped = extract_circular_region(img, radius)

	return descr_mean(img_cropped)

def descr_std_center(img, radius):
	"""
	Extract part of the center to compute its SD
	"""
	img_cropped = extract_circular_region(img, radius)
	
	return descr_std(img_cropped)

	