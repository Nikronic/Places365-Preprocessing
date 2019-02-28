import torch
from torch.utils import data

from PIL import Image

import zipfile
import zlib
import os

import errno
import os
import shutil
import zipfile
import pandas as pd


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

            :return a 3-member tuple containing input image (y_descreen) as ground truth, input image X as halftone image
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

        y_descreen = Image.open(os.path.join(self.img_dir), self.img_names[index])
        if self.transform is not None:
            y_descreen = self.transform(y_descreen)

        # TODO add halftone process here
        X = None

        # TODO add canny edge detector process here
        y_edge = None

        return X, y


import os
import tarfile

# %%
# create sample tar file to test other modules over it.
import tarfile
with tarfile.open('data.tar', 'w') as tar:
    tar.add('data/')
tar.close()


with tarfile.open('data.tar') as tf:
    for tarinfo in tf:
        if os.path.splitext(tarinfo.name)[0] == 'data/Places365_val_00000003':
            image = tf.extractfile(tarinfo)
            image = image.read()
            image = Image.open(io.BytesIO(image))