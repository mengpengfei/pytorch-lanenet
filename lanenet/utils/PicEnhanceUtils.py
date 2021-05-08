# coding: utf-8
import os
import torch
from torchvision import transforms as transforms
import numpy as np
from torchvision.transforms import functional as F
import random
import cv2
import numbers
from PIL import Image

def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

# return pil,input pil
def random_color_augmentation(gt_image):
    rv=random.random()
    if rv<0.7:
        gt_image = transforms.ColorJitter(brightness=0.05)(gt_image)
        gt_image = transforms.ColorJitter(contrast=[0.7, 1.3])(gt_image)
        gt_image = transforms.ColorJitter(saturation=[0.8, 1.2])(gt_image)
        return gt_image
    return gt_image

# return pil,input pil
def random_horizon_flip_batch_images(gt_image, gt_binary_image, gt_instance_image):
    trans=MyRandomHorizontalFlip(p=0.5)
    gt_image=trans(gt_image)
    gt_binary_image=trans(gt_binary_image)
    gt_instance_image=trans(gt_instance_image)
    return gt_image,gt_binary_image,gt_instance_image

# return pil,input pil
def random_crop(gt_image, gt_binary_image, gt_instance_image):
    rv=random.random()
    w, h = gt_image.size
    i = random.randint(0, h - 244)
    j = random.randint(0, w - 1280)
    RandomCrop = MyRandomCrop(size=(244, 1280),i=i,j=j)
    if rv<1:
        gt_image = RandomCrop(gt_image)
        # gt_image.save('./random1.jpg')
        gt_binary_image = RandomCrop(gt_binary_image)
        # gt_image.save('./random2.jpg')
        gt_instance_image = RandomCrop(gt_instance_image)
        # gt_image.save('./random3.jpg')
        return gt_image,gt_binary_image,gt_instance_image
    return gt_image,gt_binary_image,gt_instance_image

class MyRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p
        self.rv=random.random()
        # self.rv=0

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.rv < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class MyRandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant',i=0,j=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.i=i
        self.j=j
    @staticmethod
    def get_params(img, output_size,i,j):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w <= tw and h <= th:
            return 0, 0, h, w
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size,self.i,self.j)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

if __name__ == '__main__':

    # transforms.Resize((32,32))
    toPil=transforms.ToPILImage()
    # gt_imageOri = cv2.imread("D:/code/pytorch-lanenet-master/lanenet/utils/frame0270.jpg", cv2.IMREAD_COLOR)
    label_img = cv2.imread("D:/yum/tmcdata/test123/frame0270.jpg", cv2.IMREAD_COLOR)
    label_img=toPil(label_img[:,:,[2,1,0]])
    label_img.save('./random.jpg')
    print(np.asarray(label_img).shape)
    cv2.imwrite('./random1.jpg',np.asarray(label_img)[:,:,[2,1,0]])

    # gt_image=toPil(gt_imageOri)
    # imgx=np.asarray(gt_image)
    #
    # img=np.transpose(imgx,(2,0,1))
    # img=np.transpose(img,(1,2,0))
    #
    #
    # img1 = imgx.reshape(imgx.shape[2], imgx.shape[0], imgx.shape[1])
    # img1=img1.reshape(img1.shape[1],img1.shape[2],img1.shape[0])
    #
    # src7 = cv2.addWeighted(img,1.0,img1,0.3,0)
    # gt_image=random_color_augmentation(gt_image)
    # gt_image.save('./random1.jpg')
    # gt_image, gt_binary_image, label_instance_img=random_horizon_flip_batch_images(gt_image,label_img,label_img)
    # gt_image, gt_binary_image, label_instance_img=random_crop(gt_image, gt_binary_image, label_instance_img)
    #
    # resize=transforms.Resize((720,1280),interpolation=Image.NEAREST)
    # gt_image=resize(gt_image)
    # gt_binary_image=resize(gt_binary_image)
    #
    # src7 = cv2.addWeighted(cv2.cvtColor(np.asarray(gt_image),cv2.COLOR_RGB2BGR),0.8,cv2.cvtColor(np.asarray(gt_binary_image),cv2.COLOR_RGB2BGR),1,0)
    # final_img=src7
    # final_img[np.where((final_img==[255, 255, 255]).all(axis=2))] = [0,0,255]

    # gt_image, gt_binary_image, label_instance_img=random_horizon_flip_batch_images(gt_image,gt_image,label_instance_img)
    # gt_image, gt_binary_image, label_instance_img=random_crop(gt_image, gt_binary_image, label_instance_img)
    # print(np.asarray(label_instance_img).shape)

    # from lanenet.dataloader.transformers import Rescale
    # transform=transforms.Compose([Rescale((1280, 720))])
    # for i in range(0,10000):

    # label_instance_img1=label_instance_img.resize((1280, 720), Image.NEAREST)

        # resize=transforms.Resize((720,1280),interpolation=Image.NEAREST)
        # label_instance_img1=resize(label_instance_img)
        # label_instance_img=resize(label_instance_img)
        # label_instance_img=resize(label_instance_img)
        # label_instance_img.save('./random1.jpg')

    # print(np.unique(np.asarray(label_instance_img1))[1:])

    # imgx = cv2.cvtColor(np.asarray(toPil(label_instance_img[:,:,[2,1,0]])),cv2.COLOR_RGB2BGR)
    # label_instance_img=toPil(label_instance_img)
    # label_instance_img.save('./random1.jpg')
    # print(label_instance_img.shape)
    # label_instance_img=random_color_augmentation(label_instance_img)
    # label_instance_img.save('./random1.jpg')
    # gt_image, gt_binary_image, gt_instance_image=random_horizon_flip_batch_images(label_instance_img,label_instance_img,label_instance_img)
    # gt_binary_image.save('./random2.jpg')
    # gt_image, gt_binary_image, gt_instance_image=random_crop(gt_image, gt_binary_image, gt_instance_image)
    # print()
    # gt_binary_image.save('./random3.jpg')
