#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import lmdb
import cv2
import numpy as np
from torchvision import transforms
import random


def get_image_label_list(data_txt):
    gt_imgs=[]
    gt_binary_imgs=[]
    gt_instance_imgs=[]
    lines=open(data_txt).readlines()
    new = []  # 定义一个空列表，用来存储结果
    for line in lines:
        temp1 = line.strip('\n')  # 去掉每行最后的换行符'\n'
        new.append(temp1)  # 将上一步得到的列表添加到new中
    random.shuffle(new)#乱序一个列表
    for line in new:
        line_arr=line.rstrip('\n').split(' ')
        gt_imgs.append(line_arr[0])
        gt_binary_imgs.append(line_arr[1])
        gt_instance_imgs.append(line_arr[2])
    return gt_imgs,gt_binary_imgs,gt_instance_imgs


def img2lmdb(txt_path,data_name):
    # 创建数据库文件
    # env = lmdb.open(data_name)
    env = lmdb.open(data_name, max_dbs=6, map_size=int(1024*1024*1024*50))

    # 创建对应的数据库
    gt_img = env.open_db("gt_img".encode())
    gt_img_shape = env.open_db("gt_img_shape".encode())

    gt_binary_img = env.open_db("gt_binary_img".encode())
    gt_binary_img_shape = env.open_db("gt_binary_img_shape".encode())

    gt_instance_img = env.open_db("gt_instance_img".encode())
    gt_instance_shape = env.open_db("gt_instance_shape".encode())
# -----------------------val------------------------------
#     val_gt_img = env.open_db("val_gt_img")
#     val_gt_img_shape = env.open_db("val_gt_img_shape")
#
#     val_gt_binary_img = env.open_db("val_gt_binary_img")
#     val_gt_binary_img_shape = env.open_db("val_gt_binary_img_shape")
#
#     val_gt_instance_img = env.open_db("val_gt_instance_img")
#     val_gt_instance_img_shape = env.open_db("val_gt_instance_img_shape")

    gt_imgs,gt_binary_imgs,gt_instance_imgs = get_image_label_list(txt_path)
    #print(gt_binary_imgs)
    # val_gt_imgs,val_gt_binary_imgs,val_gt_instance_imgs = get_image_label_list('val_txt_path')
    # 把图像数据写入到LMDB中
    with env.begin(write=True) as txn:
        for idx, path in enumerate(gt_imgs):
            print("{} {}".format(idx, path))
            data = cv2.imread(path, cv2.IMREAD_COLOR)
            print(path)
            txn.put(str(idx).encode(), data, db=gt_img)
            txn.put(str(idx).encode(),"".join(str(data.shape)).replace('(','').replace(')','').encode(), db=gt_img_shape)

        for idx, path in enumerate(gt_binary_imgs):
            print("{} {}".format(idx, path))
            data = cv2.imread(path, cv2.IMREAD_COLOR)
            txn.put(str(idx).encode(), data, db=gt_binary_img)
            txn.put(str(idx).encode(),"".join(str(data.shape)).replace('(','').replace(')','').encode(), db=gt_binary_img_shape)

        for idx, path in enumerate(gt_instance_imgs):
            print("{} {}".format(idx, path))
            data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            txn.put(str(idx).encode(), data, db=gt_instance_img)
            txn.put(str(idx).encode(),"".join(str(data.shape)).replace('(','').replace(')','').encode(), db=gt_instance_shape)
        # txn.commit()
    env.close()
if __name__ == '__main__':
    # get_image_label_list('D:/yum/tmcdata/test123/val.txt')
    img2lmdb('/workspace/mogo_data/index/train.txt','train')
    # env = lmdb.open('./val', max_dbs=6, map_size=int(1024*1024*1024*8), readonly=True)
    # 创建对应的数据库
    # gt_img = env.open_db("gt_img".encode())
    # gt_img_shape = env.open_db("gt_img_shape".encode())
    #
    # gt_binary_img = env.open_db("gt_binary_img".encode())
    # gt_binary_img_shape = env.open_db("gt_binary_img_shape".encode())
    #
    # gt_instance_img = env.open_db("gt_instance_img".encode())
    # gt_instance_shape = env.open_db("gt_instance_shape".encode())
    #
    # txn = env.begin()
    # _length = txn.stat(db=gt_img)["entries"]
    #
    # a=np.frombuffer(txn.get('0'.encode(), db=gt_img),'uint8')
    # abcd=str(txn.get('0'.encode(), db=gt_img_shape).decode()).replace(' ','').split(',')
    # img = a.reshape(int(abcd[0]), int(abcd[1]),int(abcd[2]))
    #
    # toPIL = transforms.ToPILImage()
    # imgori = toPIL(img)
    # imgori.save('./randomxx.jpg')
    # cv2.imwrite('./random.jpg',np.asarray(imgori))
    # print()

    # np.reshape(a,)