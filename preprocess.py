import torch
from torch.utils import data

class Dataset(data.Dataset):
    """
    Return Dataset class representing our data set
    """
    def __int__(self, list_IDs, labels):
        """
        Initialize data set as a list of IDs corresponding to each item of data set and labels of each data

        Args:
            list_IDs: a list of IDs for each data point in data set
            labels: label of an item in data set with respect to the ID
        """

        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.list_IDs)

    def __getitem__(self, item):
        """
        Generate one item of data set. Here we apply our preprocessing things like halftone styles and subtractive color process using CMYK color model etc. (See the paper for operations)

        :param item: index of item in IDs list

        :return: a sample of data
        """
        ID = self.list_IDs[item]

        # Code to load data
        X = None #
        y = None #

        return X, y

