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
        Generate one item of data set. Here we apply our preprocessing things like halftone styles and
        subtractive color process using CMYK color model etc. (See the paper for operations)

        :param item: index of item in IDs list

        :return: a sample of data
        """
        ID = self.list_IDs[item]

        # Code to load data
        X = None #
        y = None #

        return X, y


# Enable CUDA
has_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if has_cuda else "cpu")
# cudnn.benchmark = True


# Datasets
partition = None # IDs
labels = None # labels

# Parameters
params = {
    'batch_size':64,
    'shuffle': True,
    'num_workers':4
}
max_epochs = 100

# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

# Do same for validation


# Loop over epochs to train
for epoch in range(max_epochs):
    for local_batch, local_label in training_generator:
        local_batch, local_label = local_batch.to(device), local_label.to(device)
        # Model

# Validate model
with torch.set_grad_enabled(False):
    pass






import zipfile
import zlib
import os

import errno
import os
import shutil
import zipfile



#%%

def generate_partition(zip_dir):
    """
    src = open(zip_dir, 'rb')
    zf = zipfile.ZipFile(src)
    for idx,m in enumerate(zf.infolist()):
        if idx != 0:
            src.read(30)
            decomp = zlib.decompressobj(-15)
            out = open('data/'+m.filename, "wb")
            result = decomp.decompress(src.read(m.compress_size))
            out.write(result)
            result = decomp.flush()
            out.write(result)
            print(out)
            out.close()
        else:
            pass

    zf.close()
    src.close()
    """

    TARGETDIR = 'small_extract/'
    with open(zip_dir, "rb") as zipsrc:
        zfile = zipfile.ZipFile(zipsrc)
        for member in zfile.infolist():
            target_path = os.path.join(TARGETDIR, member.filename)
            if target_path.endswith('/'):  # folder entry, create
                try:
                    os.makedirs(target_path)
                except:
                    raise

            with open(target_path, 'wb') as outfile, zfile.open(member) as infile:
                shutil.copyfileobj(infile, outfile)
                print(infile)
                print(outfile)


generate_partition('data/small.zip')


import ctypes, sys

windows = [{
    'script': "preprocess.py",
    'uac_info': "requireAdministrator",
},]

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False