from torchvision.transforms import ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, RandomHorizontalFlip
from torchvision import transforms
from preprocess import *
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

custom_transforms = transforms.Compose([ToPILImage(),
                                        RandomResizedCrop(size=224, scale=(0.8, 1.2)),
                                        RandomRotation(degrees=(-30, 30)),
                                        RandomHorizontalFlip(p=0.5),
                                        RandomNoise(p=0.5, mean=0, std=0.1),
                                        ToTensor()])

# TODO out input images based on custom Dataset class, are PIL images, so they are [0,1]. Now we have to transform them into [-1,1]. when we should do it??? The idea is PIL images are normalized already and ToTensor also normalizing them to [0, 1], So i remove normalizing layer!

# https://discuss.pytorch.org/t/what-does-pil-images-of-range-0-1-mean-and-how-do-we-save-images-as-that-format/2103

train_dataset = PlacesDataset(txt_path='data/filelist.txt',
                              img_dir='data.tar',
                              transform=None)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=2)  # TODO change to desired one on colab

# class Dataset:
#     def __int__(self, txt_path='data/filelist.txt', img_dir='data.tar', transform=None):
#
#         """
#         Initialize data set as a list of IDs corresponding to each item of data set
#
#         :param img_dir: path to the main tar file of all of images
#         :param txt_path: a text file containing names of all of images line by line
#         :param transform: apply some transforms like cropping, rotating, etc on input image
#
#         :return a 3-value dict containing input image (y_descreen) as ground truth, input image X as halftone image
#                 and edge-map (y_edge) of ground truth image to feed into the network.
#         """
#
#         df = pd.read_csv(txt_path, sep=' ', index_col=0)
#         self.img_names = df.index.values
#         self.txt_path = txt_path
#         self.img_dir = img_dir
#         self.transform = transform
#
#     @staticmethod
#     def canny_edge_detector(image):
#         """
#         Returns a binary image with same size of source image which each pixel determines belonging to an edge or not.
#
#         :param image: PIL image
#         :return: Binary numpy array # TODO check if conversion to PIL image from binary numpy is necessary.
#         """
#         image = np.array(image)
#         image = color.rgb2grey(image)
#         edges = feature.canny(image, sigma=1)  # TODO: the sigma hyper parameter value is not defined in the paper.
#         return edges * 1
#
#
#     def get_image_by_name(self, name):
#         """
#         gets a image by a name gathered from file list csv file
#
#         :param name: name of targeted image
#         :return: a PIL image
#         """
#         image = None
#         with tarfile.open(self.img_dir) as tf:
#             for tarinfo in tf:
#                 if tarinfo.name == name:
#                     image = tf.extractfile(tarinfo)
#                     image = image.read()
#                     image = Image.open(io.BytesIO(image))
#         return image
#
#     def __len__(self):
#         """
#         Return the length of data set using list of IDs
#
#         :return: number of samples in data set
#         """
#         return len(self.img_names)
#
#     def __getitem__(self, index):
#         """
#         Generate one item of data set. Here we apply our preprocessing things like halftone styles and
#         subtractive color process using CMYK color model, generating edge-maps, etc.
#
#         :param index: index of item in IDs list
#
#         :return: a sample of data as a dict
#         """
#
#         y_descreen = self.get_image_by_name(self.img_names[index])
#
#         # generate halftone image
#         X = generate_halftone(y_descreen)
#
#         # generate edge-map
#         y_edge = self.canny_edge_detector(y_descreen)
#
#         if self.transform is not None:
#             X = self.transform(X)
#
#         sample = {'X':X,
#                   'y_descreen':y_descreen,
#                   'y_edge':y_edge}
#
#         return sample

# TODO just uncomment above code to see how much fucked up is PYTHON. SAME CLASSES, DIFF NAMES, DIFF RESULTS!!!!
