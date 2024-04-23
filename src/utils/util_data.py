import torch
import torchvision.transforms as T
import numpy as np
import cv2
import os
import imutils
import random
from torchvision.io import read_image
from scipy.ndimage import shift
import nrrd


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_img(img_path, format):
    if format == "nrrd":
        img, _ = nrrd.read(img_path)
        return img
    if format == "jpg":
        img = read_image(img_path)
        img = T.functional.rgb_to_grayscale(img, 1)
        img = torch.squeeze(img, 0)
        img = img.numpy()
        return img


def pad(img, r_l, pad_value=0):
    max_size = max(img.shape[1], img.shape[0])
    min_size = min(img.shape[1], img.shape[0])
    num_zeros = max_size - min_size
    if img.shape[0] > img.shape[1]:
        if r_l == "R":
            img = np.pad(img, ((0, 0), (num_zeros, 0)), 'constant', constant_values=pad_value)  # ((top, bottom), (left, right))
        if r_l == "L":
            img = np.pad(img, ((0, 0), (0, num_zeros)), 'constant', constant_values=pad_value)
    if img.shape[1] > (img.shape[0]):
        img = np.pad(img, ((0, num_zeros), (0, 0)), 'constant', constant_values=pad_value)
    return img


def clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def augmentation(img_x, img_y):
    # shift
    r = random.randint(0, 100)
    if r > 70:
        shift_perc = 0.1
        r1 = random.randint(-int(shift_perc * img_x.shape[0]), int(shift_perc * img_x.shape[0]))
        r2 = random.randint(-int(shift_perc * img_x.shape[1]), int(shift_perc * img_x.shape[1]))
        img_x = shift(img_x, [r1, r2], mode='nearest')
        img_y = shift(img_y, [r1, r2], mode='nearest')
    # zoom
    r = random.randint(0, 100)
    if r > 70:
        zoom_perc = 0.1
        zoom_factor = random.uniform(1 - zoom_perc, 1 + zoom_perc)
        img_x = clipped_zoom(img_x, zoom_factor=zoom_factor)
        img_y = clipped_zoom(img_y, zoom_factor=zoom_factor)
    # flip
    r = random.randint(0, 100)
    if r > 70:
        img_x = cv2.flip(img_x, 1)
        img_y = cv2.flip(img_y, 1)
    # rotation
    r = random.randint(0, 100)
    if r > 70:
        max_angle = 15
        r = random.randint(-max_angle, max_angle)
        img_x = imutils.rotate(img_x, r)
        img_y = imutils.rotate(img_y, r)
    return img_x, img_y


def contrast_stretching(img, ww, wc):
    pixel_min = wc - ww / 2
    pixel_max = wc + ww / 2
    img = np.where(img > pixel_min, img, pixel_min)
    img = np.where(img < pixel_max, img, pixel_max)
    return (img)


def loader_public_dataset(img_x_path, img_y_path, r_l, img_dim, do_augmentation, step="train"):
    # Img
    img_x = load_img(img_path=img_x_path, format="jpg")
    img_y = load_img(img_path=img_y_path, format="jpg")

    # Pad
    img_x = pad(img_x, r_l)
    img_y = pad(img_y, r_l)

    # Norm
    img_x = (img_x - np.amin(img_x)) / (np.amax(img_x) - np.amin(img_x))
    img_y = (img_y - np.amin(img_y)) / (np.amax(img_y) - np.amin(img_y))

    # Resize
    img_x = cv2.resize(img_x, dsize=(img_dim, img_dim))
    img_y = cv2.resize(img_y, dsize=(img_dim, img_dim))

    # Augmentation
    if do_augmentation:
        if step == "train":
            img_x, img_y = augmentation(img_x, img_y)
    # To Tensor
    img_x = torch.Tensor(img_x)
    img_y = torch.Tensor(img_y)
    img_x = torch.unsqueeze(img_x, dim=0)
    img_y = torch.unsqueeze(img_y, dim=0)
    return img_x, img_y


# loader con contrast stretching per FPUCBM dataset
def loader(img_x_path, img_y_path, r_l, img_dim, ww_x, wc_x, ww_y, wc_y, do_augmentation, step="train"):
    # Img
    img_x = load_img(img_path=img_x_path, format="nrrd")
    img_y = load_img(img_path=img_y_path, format="nrrd")

    # Pad
    img_x = pad(img_x, r_l)
    img_y = pad(img_y, r_l)

    # Contrast Stretching
    img_x = contrast_stretching(img_x, ww_x, wc_x)
    img_y = contrast_stretching(img_y, ww_y, wc_y)

    # Norm
    img_x = (img_x - np.amin(img_x)) / (np.amax(img_x) - np.amin(img_x))
    img_y = (img_y - np.amin(img_y)) / (np.amax(img_y) - np.amin(img_y))

    # Resize
    img_x = cv2.resize(img_x, dsize=(img_dim, img_dim))
    img_y = cv2.resize(img_y, dsize=(img_dim, img_dim))

    # Augmentation
    if do_augmentation:
        if step == "train":
            img_x, img_y = augmentation(img_x, img_y)

    # To Tensor
    img_x = torch.Tensor(img_x)
    img_y = torch.Tensor(img_y)
    img_x = torch.unsqueeze(img_x, dim=0)
    img_y = torch.unsqueeze(img_y, dim=0)
    return img_x, img_y


# ImgDataset per public dataset
class ImgDataset_public_dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, cfg_data, step, do_augmentation):
        'Initialization'
        self.step = step
        self.img_dir = cfg_data["img_dir"]
        self.data = data
        # Dim
        self.img_dim = cfg_data["img_dim"]
        self.do_augmentation = do_augmentation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        img_x_file = row.img_x
        img_y_file = row.img_y
        r_l = row.r_l

        # Load data
        img_x_path = os.path.join(self.img_dir, img_x_file).replace("\\","/")
        img_y_path = os.path.join(self.img_dir, img_y_file).replace("\\", "/")
        img_x, img_y = loader_public_dataset(img_x_path=img_x_path, img_y_path=img_y_path, r_l=r_l,
                                             img_dim=self.img_dim, do_augmentation=self.do_augmentation, step=self.step)
        return img_x, img_y, id


# ImgDataset per FPUCBM dataset
class ImgDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, cfg_data, step, do_augmentation):
        'Initialization'
        self.step = step
        self.img_dir = cfg_data["img_dir"]
        self.data = data
        # Dim
        self.img_dim = cfg_data["img_dim"]
        # Augmentation
        self.do_augmentation = do_augmentation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        img_x_file = row.img_x
        img_y_file = row.img_y
        r_l = row.r_l
        wc_x = row.wc_x
        ww_x = row.ww_x
        wc_y = row.wc_y
        ww_y = row.ww_y
        # Load data
        img_x_path = os.path.join(self.img_dir, img_x_file).replace("\\","/")
        img_y_path = os.path.join(self.img_dir, img_y_file).replace("\\", "/")
        img_x, img_y = loader(img_x_path=img_x_path, img_y_path=img_y_path, r_l=r_l, img_dim=self.img_dim, ww_x=ww_x,
                              wc_x=wc_x, ww_y=ww_y, wc_y=wc_y, do_augmentation=self.do_augmentation, step=self.step)
        return img_x, img_y, id
