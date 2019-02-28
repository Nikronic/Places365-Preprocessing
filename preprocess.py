import torch
from torch.utils import data

from PIL import Image
from skimage import feature, color
import numpy as np

import tarfile
import os
import io
import pandas as pd

from Halftoning.halftone import generate_halftone

import matplotlib.pyplot as plt # TODO remove after tests completed!



class Dataset(data.Dataset):
    """
    Return Dataset class representing our data set
    """
    def __int__(self, txt_path, img_dir, transform):
        """
        Initialize data set as a list of IDs corresponding to each item of data set and labels of each data

        Args:
            :param img_dir: path to the main tar file of all of images
            :param txt_path: a text file containing names of all of images line by line
            :param transform: apply some transforms like cropping, rotating, etc on input image

            :return a 3-value tuple containing input image (y_descreen) as ground truth, input image X as halftone image
                    and edge-map (y_edge) of ground truth image to feed into the network.
        """

        df = pd.read_csv(txt_path, sep=' ', index_col=0)
        self.img_names = df.index.values
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set. Here we apply our preprocessing things like halftone styles and
        subtractive color process using CMYK color model, generating edge-maps, etc.

        :param index: index of item in IDs list

        :return: a sample of data
        """

        y_descreen = get_image_by_name(self.img_dir, self.img_names[index])

        # generate halftone image
        X = halftone.generate_halftone(y_descreen)

        # generate edge-map
        y_edge = canny_edge_detector(y_descreen)

        if self.transform is not None:
            X = self.transform(X)

        return X, y_descreen, y_edge

    @staticmethod
    def canny_edge_detector(image):
        image = np.array(image)
        image = color.rgb2grey(image)
        edges = feature.canny(image, sigma=3)  # TODO: the sigma hyper parameter value is not defined in the paper.
        return edges*1

    def get_image_by_name(self, name):
        image = None
        with tarfile.open(self.img_dir) as tf:
            for tarinfo in tf:
                if os.path.splitext(tarinfo.name)[0] == name:
                    image = tf.extractfile(tarinfo)
                    image = image.read()
                    image = Image.open(io.BytesIO(image))
        return image

# %%
# create sample tar file to test other modules over it.

with tarfile.open('data.tar', 'w') as tar:
    tar.add('data/')



def canny_edge_detector(image):
    image = np.array(image)
    image = color.rgb2grey(image)
    edges = feature.canny(image, sigma=1)  # TODO: the sigma hyper parameter value is not defined in the paper.
    return edges

def geti(path, name):
    with tarfile.open(path) as tf:
        for tarinfo in tf:
            if os.path.splitext(tarinfo.name)[0] == name:
                image = tf.extractfile(tarinfo)
                image = image.read()
                image = Image.open(io.BytesIO(image))
    return image

i = geti('data.tar', 'data/Places365_val_00000001')
ie = canny_edge_detector(i)

ih = generate_halftone(i)

plt.imshow(ih)
plt.show()

