from torchvision.transforms import *
import torch

from preprocess import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

custom_transforms = transforms.Compose([ToPILImage(),
                                        RandomResizedCrop(size=224, scale=(0.8, 1.2)),
                                        RandomRotation(degrees=(-30, 30)),
                                        RandomHorizontalFlip(p=0.5),
                                        RandomNoise(p=0.5, mean=0, std=0.1),
                                        ToTensor()])

# TODO out input images based on custom Dataset class, are PIL images, so they are [0,1]. Now we have to transform them into [-1,1]. when we should do it??? The idea is PIL images are normalized already and ToTensor also normalizing them to [0, 1], So i remove normalizing layer!

# https://discuss.pytorch.org/t/what-does-pil-images-of-range-0-1-mean-and-how-do-we-save-images-as-that-format/2103

train_dataset = HalftoneDataset('data/filelist.txt'
                                , 'data.tar',
                                custom_transforms)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=8)
