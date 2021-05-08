import lmdb
from torch.utils.data import Dataset, DataLoader
from lanenet import config
import numpy as np
from torchvision import transforms
from lanenet.utils import PicEnhanceUtils
from PIL import Image

class LaneDataSet(Dataset):

    def __init__(self, dataset, n_labels=config.n_labels, transform=None):
        env = lmdb.open(dataset, max_dbs=6, map_size=int(1024*1024*1024*8), readonly=True)
        # 创建对应的数据库
        self.gt_img_db = env.open_db("gt_img".encode())
        self.gt_img_shape_db = env.open_db("gt_img_shape".encode())

        self.gt_binary_img_db = env.open_db("gt_binary_img".encode())
        self.gt_binary_img_shape_db = env.open_db("gt_binary_img_shape".encode())

        self.gt_instance_img_db = env.open_db("gt_instance_img".encode())
        self.gt_instance_img_shape_db = env.open_db("gt_instance_shape".encode())
        self.n_labels = n_labels

        self.txn = env.begin()
        self._length = self.txn.stat(db=self.gt_img_db)["entries"]

    def _split_instance_gt(self, label_instance_img):
        # number of channels, number of unique pixel values, subtracting no label
        # adapted from here https://github.com/nyoki-mtl/pytorch-discriminative-loss/blob/master/src/dataset.py
        no_of_instances = self.n_labels
        ins = np.zeros((no_of_instances, label_instance_img.shape[0], label_instance_img.shape[1]))
        for _ch, label in enumerate(np.unique(label_instance_img)[1:]):
            ins[_ch, label_instance_img == label] = 1
        return ins

    def __getitem__(self, idx):
        idx = str(idx).encode()

        gt_img_buf = self.txn.get(idx, db=self.gt_img_db)
        gt_img_array = np.frombuffer(gt_img_buf, dtype=np.uint8)
        gt_img_list=str(self.txn.get(idx, db=self.gt_img_shape_db).decode()).replace(' ','').split(',')
        img=gt_img_array.reshape(int(gt_img_list[0]), int(gt_img_list[1]), int(gt_img_list[2]))


        gt_binary_img_buf = self.txn.get(idx, db=self.gt_binary_img_db)
        gt_binary_img_array = np.frombuffer(gt_binary_img_buf, dtype=np.uint8)
        gt_binary_img_list=str(self.txn.get(idx, db=self.gt_binary_img_shape_db).decode()).replace(' ','').split(',')
        label_img=gt_binary_img_array.reshape(int(gt_binary_img_list[0]), int(gt_binary_img_list[1]),int(gt_binary_img_list[2]))

        gt_instance_img_buf = self.txn.get(idx, db=self.gt_instance_img_db)
        gt_instance_img_array = np.frombuffer(gt_instance_img_buf, dtype=np.uint8)
        gt_instance_img_list=str(self.txn.get(idx, db=self.gt_instance_img_shape_db).decode()).replace(' ','').split(',')
        label_instance_img=gt_instance_img_array.reshape(int(gt_instance_img_list[0]), int(gt_instance_img_list[1]))

        toPil=transforms.ToPILImage()

        # img=toPil(img[:,:,[2,1,0]])
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
        label_instance_img = self._split_instance_gt(label_instance_img)

        # img = img.reshape(img.shape[2], img.shape[0], img.shape[1])

        img=np.transpose(img,(2,0,1))


        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        return img, label_binary, label_instance_img

    def __len__(self):
        return self._length