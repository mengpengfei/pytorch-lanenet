# coding: utf-8

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

from lanenet.utils import PicEnhanceUtils

import random
from lanenet import config

class LaneDataSet(Dataset):
    def __init__(self, dataset, n_labels=config.n_labels, transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform
        self.n_labels = n_labels

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])

        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = zip(*c)

    def _split_instance_gt(self, label_instance_img):
        # number of channels, number of unique pixel values, subtracting no label
        # adapted from here https://github.com/nyoki-mtl/pytorch-discriminative-loss/blob/master/src/dataset.py
        no_of_instances = self.n_labels
        ins = np.zeros((no_of_instances, label_instance_img.shape[0], label_instance_img.shape[1]))
        for _ch, label in enumerate(np.unique(label_instance_img)[1:]):
            ins[_ch, label_instance_img == label] = 1
        return ins

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        # load all
        img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR)

        label_instance_img = cv2.imread(self._gt_label_instance_list[idx], cv2.IMREAD_UNCHANGED)

        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)
        # print("------------------------------------------------------------------")
        # print(img.size())
        # print("------------------------------------------------------------------")
        # optional transformations

        toPil=transforms.ToPILImage()

        img=toPil(img[:,:,[2,1,0]])
        label_instance_img=toPil(label_instance_img)
        label_img=toPil(label_img[:,:,[2,1,0]])

        img=PicEnhanceUtils.random_color_augmentation(img)
        img,label_img,label_instance_img=PicEnhanceUtils.random_horizon_flip_batch_images(img,label_img,label_instance_img)
        img,label_img,label_instance_img=PicEnhanceUtils.random_crop(img,label_img,label_instance_img)

        resize=transforms.Resize((720,1280),interpolation=Image.NEAREST)
        img=resize(img)
        label_img=resize(label_img)
        label_instance_img=resize(label_instance_img)

        # if self.transform:
        img=np.asarray(img)
        label_img=np.asarray(label_img)
        label_instance_img=np.asarray(label_instance_img)

        # extract each label into separate binary channels
        # print(self._gt_label_instance_list[idx])
        label_instance_img = self._split_instance_gt(label_instance_img)


        # reshape for pytorch
        # tensorflow: [height, width, channels]
        # pytorch: [channels, height, width]
        # print("------------------------------------------------------------------")
        # print(img.size())
        # print("------------------------------------------------------------------")
        # img = img.reshape(img.shape[2], img.shape[0], img.shape[1])

        img=np.transpose(img,(2,0,1))

        # print("------------------------------------------------------------------")
        # print(img.size())
        # print("------------------------------------------------------------------")


        # print("///////////////////////////////////////////////////")
        # print(img.shape)
        # print(label_img.shape)
        # print("///////////////////////////////////////////////////")

        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        return img, label_binary, label_instance_img
