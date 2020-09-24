import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

class SynthesisDataset(Dataset):
    '''
    Dataset class for synthesized dataset
    Zhe Zhu, 2020/05/17
    '''
    def __init__(self,dataset_folder_list,sample_num_list,scale=1.0):
        self.dataset_folder_list = dataset_folder_list
        self.sample_num_list = sample_num_list
        self.scale = scale

        self.ids = []
        for i_d,dataset_folder in enumerate(dataset_folder_list):
            sample_num = sample_num_list[i_d]
            for i_s in range(sample_num):
                id = os.path.join(dataset_folder, "{:05d}".format(i_s))
                self.ids.append(id)
        print("Creating dataset with {0} samples".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)


    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        id = self.ids[i]
        img_file = id + ".png"
        mask_file = id + "_mask.png"

        img = Image.open(img_file)
        mask = Image.open(mask_file)

        img = self.preprocess(img,self.scale)
        mask = self.preprocess(mask,self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

class DukeDataset(Dataset):
    '''
    Duke CT&MRI Segmentation Dataset
    '''

    def __init__(self,series_folder_list,newW,newH):
        self.newW = newW
        self.newH = newH
        self.img_file_list = []
        self.mask_file_list = []

        for series_folder in series_folder_list:
            img_folder = os.path.join(series_folder,'img')
            mask_folder = os.path.join(series_folder,'mask')
            img_file_list = glob(img_folder+'/*')
            img_num = len(img_file_list)

            for i in range(img_num):
                img_file = os.path.join(img_folder,'{:04d}.png'.format(i))
                mask_file = os.path.join(mask_folder,'{:04d}.png'.format(i))
                self.img_file_list.append(img_file)
                self.mask_file_list.append(mask_file)

        print("Creating dataset with {0} samples".format(len(self.img_file_list)))

    def __len__(self):
        return len(self.img_file_list)


    @classmethod
    def preprocess(cls, pil_img,newW, newH):
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)
        img_nd = img_nd.astype(np.float64)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        img_file = self.img_file_list[i]
        mask_file = self.mask_file_list[i]

        img = Image.open(img_file)
        mask = Image.open(mask_file)

        img = self.preprocess(img,self.newW,self.newH)
        mask = self.preprocess(mask,self.newW,self.newH)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}