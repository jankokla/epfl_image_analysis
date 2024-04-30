import os

import numpy as np
import pandas as pd
from PIL import Image

from matplotlib import image as mpimg
from torch.utils.data import Dataset

from project.scripts.config import LABELS_TO_IDS


class CoinDataset(Dataset):
    """
    Dataset class for preprocessing and delivering satellite images.

    :param image_dir: local absolute path of training images directory
    :param transform: Albumentations obj for transforms
    :param preprocess: to adjust the transforms to encoder
    """

    def __init__(self, image_dir, transform=None, preprocess=None):
        # read images in
        files = os.listdir(image_dir)
        image_files = [os.path.join(image_dir, f) for f in files if f.endswith('.png')]
        self.images = [mpimg.imread(path) for path in image_files]

        labels_path = os.path.join(image_dir, 'labels.csv')

        self.labels = None
        if 'labels.csv' in files:
            files.remove('labels.csv')
            raw_labels = pd.read_csv(labels_path, index_col=1).loc[files].label.tolist()
            self.labels = [LABELS_TO_IDS[label] for label in raw_labels]

        self.transform = transform
        self.preprocess = preprocess

    def __getitem__(self, i):
        """Getter for providing images to DataLoader."""
        image = self.images[i]
        label = self.labels[i]

        if self.transform:
            # apply same transformation to image and mask
            # NB! This must be done before converting to Pytorch format
            image = self.transform(image=image)['image']

        # apply preprocessing to adjust to encoder
        if self.preprocess:
            image = Image.fromarray(np.uint8(image * 255))
            image = self.preprocess(image)

        return image, label
