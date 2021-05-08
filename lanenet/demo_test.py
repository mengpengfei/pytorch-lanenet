import argparse
import time

import torch
from torch.autograd import Variable
import numpy as np
import cv2
from torchvision import transforms
from lanenet.dataloader.transformers import Rescale
from lanenet.model.model import LaneNet
from lanenet.utils.postprocess import embedding_post_process
import torch.nn as nn
import os
DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = '7'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/demo.jpg", help="Path to demo img")
    parser.add_argument("--model_path", '-w', type=str, help="Path to model weights")
    # parser.add_argument("--band_width", '-b', type=float, default=1.5, help="Value of delta_v")
    # parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    model_path = args.model_path

    # global best_epoch
    # global args
    gpu = True
    if not torch.cuda.is_available():
        gpu = False

    model = LaneNet()
    model.to(DEVICE)

    if gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    model.eval()

    transform=transforms.Compose([Rescale((1280, 720))])
    imgori = cv2.imread(img_path, cv2.IMREAD_COLOR)
    imgori=transform(imgori)
    toPIL = transforms.ToPILImage()
    img = np.asarray(toPIL(imgori[:,:,[2,1,0]]))
    img=np.transpose(img,(2,0,1))
    img = np.expand_dims(img,0)
    print(img.shape)
    imgdata=Variable(torch.from_numpy(img)).type(torch.FloatTensor).to(DEVICE)
    print(imgdata.size())
    output=model(imgdata)

    embedding = output['instance_seg_logits']
    embedding = embedding.detach().cpu().numpy()
    embedding = np.transpose(embedding[0], (1, 2, 0))

    bin_seg_pred=output["binary_seg_pred"][0][0].detach().cpu().numpy()

    img = cv2.cvtColor(imgori, cv2.COLOR_RGB2BGR)
    seg_img = np.zeros_like(img)

    lane_seg_img = embedding_post_process(embedding, bin_seg_pred, band_width=3, max_num_lane=6)
    color = np.array([
        [255, 125, 0],
        [0, 255, 0],
        [0, 0, 255],
        [0, 255, 255],
        [255, 0, 0],
        [255, 255, 0]], dtype='uint8')

    for i, lane_idx in enumerate(np.unique(lane_seg_img)):
        if lane_idx==0:
            continue
        seg_img[lane_seg_img == lane_idx] = color[i-1]
    img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)

    cv2.imwrite("demo/demo_result.jpg", img)

    # if args.visualize:
    if True:
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
