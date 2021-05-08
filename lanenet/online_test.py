import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
from lanenet.model.model import LaneNet, compute_loss
import cv2
# import torch.nn.functional as F
from torchvision.transforms import functional as F
import time
from torchvision import transforms
from lanenet.dataloader.transformers import Rescale
import os
DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    model_path = './checkpoints-combine/773_checkpoint.pth'
    gpu = True
    if not torch.cuda.is_available():
        gpu = False
    model = LaneNet()
    model.to(DEVICE)
    # if gpu:
    #     model = model.cuda()
    # print('loading pretrained model from %s' % model_path)
    if gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    
    #if gpu:
    #    model.load_state_dict(torch.load(model_path))
    #else:
    #    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    picPath="/workspace/mogo_data/index/test"
    video=cv2.VideoWriter('./vedio/test1.avi',cv2.VideoWriter_fourcc(*'MJPG'),25,(1280, 720))
    for file in os.listdir(picPath):
        #file="frame0270.jpg"
        toPIL = transforms.ToPILImage()
        #transform=transforms.Compose([Rescale((720, 1280))])
        file_path = os.path.join(picPath, file)
        imgori = cv2.imread(file_path, cv2.IMREAD_COLOR)

        #img0=np.asarray(F.crop(toPIL(imgori[:,:,[2,1,0]]), imgori.shape[0]-240,0, 240, imgori.shape[1]))
        #img0 = transform(imgori)
        img0=imgori
        img = img0.reshape(img0.shape[2], img0.shape[0], img0.shape[1])
        img = np.expand_dims(img,0)
        print(img.shape)
        imgdata=Variable(torch.from_numpy(img)).type(torch.FloatTensor).to(DEVICE)
        output=model(imgdata)
        binary_seg_pred=output["binary_seg_pred"]
        binary_seg_pred = binary_seg_pred.squeeze(0)
        binary_seg_pred1=binary_seg_pred.to(torch.float32).cpu()
        pic = toPIL(binary_seg_pred1)
        imgx = cv2.cvtColor(np.asarray(pic),cv2.COLOR_RGB2BGR)
        imgx[np.where((imgx!=[0, 0, 0]).all(axis=2))] = [255,255,255]
        src7 = cv2.addWeighted(img0,0.8,imgx,1,0)
        final_img=src7
        # final_img=cv2.resize(src7,(1280, 720))
        final_img[np.where((final_img==[255, 255, 255]).all(axis=2))] = [0,0,255]
        #cv2.imwrite('./random.jpg',final_img)
        video.write(final_img)
        #break
        print('-----------------------runing---------------------------------')
    print('-----------------------end---------------------------------')
    video.release()




