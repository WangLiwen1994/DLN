import os
import random
import sys
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps, ImageEnhance

path = os.getcwd()
sys.path.append(path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((ty, tx, ty + tp, tx + tp))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    # img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    # info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        # img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            # img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            # img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target = load_img(self.hr_image_filenames[index])
        # name = self.hr_image_filenames[index]
        # lr_name = name[:25]+'LR/'+name[28:-4]+'x4.png'
        # lr_name = name[:18] + 'LR_4x/' + name[21:]
        input = load_img(self.lr_image_filenames[index])

        # target = ImageOps.equalize(target)
        # input_eq = ImageOps.equalize(input)
        # target = ImageOps.equalize(target)
        #         # input = ImageOps.equalize(input)
        input, target, = get_patch(input, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            img_in, img_tar, _ = augment(input, target)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar

    def __len__(self):
        return len(self.hr_image_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            # input = self.transform(input)
            bicubic = self.transform(bicubic)

        return bicubic, file

    def __len__(self):
        return len(self.image_filenames)


class Lowlight_DatasetFromVOC(data.Dataset):
    def __init__(self, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(Lowlight_DatasetFromVOC, self).__init__()
        self.imgFolder = "datasets/VOC2007/JPEGImages"
        self.image_filenames = [join(self.imgFolder, x) for x in listdir(self.imgFolder) if is_image_file(x)]

        self.image_filenames = self.image_filenames
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        ori_img = load_img(self.image_filenames[index])  # PIL image
        width, height = ori_img.size
        ratio = min(width, height) / 384

        newWidth = int(width / ratio)
        newHeight = int(height / ratio)
        ori_img = ori_img.resize((newWidth, newHeight), Image.ANTIALIAS)

        high_image = ori_img

        ## color and contrast *dim*
        color_dim_factor = 0.3 * random.random() + 0.7
        contrast_dim_factor = 0.3 * random.random() + 0.7
        ori_img = ImageEnhance.Color(ori_img).enhance(color_dim_factor)
        ori_img = ImageEnhance.Contrast(ori_img).enhance(contrast_dim_factor)

        ori_img = cv2.cvtColor((np.asarray(ori_img)), cv2.COLOR_RGB2BGR)  # cv2 image
        ori_img = (ori_img.clip(0, 255)).astype("uint8")
        low_img = ori_img.astype('double') / 255.0

        # generate low-light image
        beta = 0.5 * random.random() + 0.5
        alpha = 0.1 * random.random() + 0.9
        gamma = 3.5 * random.random() + 1.5
        low_img = beta * np.power(alpha * low_img, gamma)

        low_img = low_img * 255.0
        low_img = (low_img.clip(0, 255)).astype("uint8")
        low_img = Image.fromarray(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB))

        img_in, img_tar = get_patch(low_img, high_image, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            img_in, img_tar, _ = augment(img_in, img_tar)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar

    def __len__(self):
        return len(self.image_filenames)
