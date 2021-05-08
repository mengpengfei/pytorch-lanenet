import time

import torch
from torch.autograd import Variable
import numpy as np
import cv2
from torchvision import transforms
from lanenet.dataloader.transformers import Rescale
from lanenet.model.model import LaneNet
import torch.nn as nn
import os
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def compose_img(image_data, out, i=0):
    oridata=image_data[i].cpu().numpy()
    val_gt=oridata.transpose(1, 2, 0).astype(np.uint8)
    #val_gt=oridata.reshape(oridata.shape[1],oridata.shape[2],oridata.shape[0]).astype(np.uint8)
    # val_gt = (.transpose(1, 2, 0)).astype(np.uint8)
    predata=out[i].squeeze(0).cpu().numpy()
    val_pred=predata.transpose(0, 1)*255
    # val_pred=predata.reshape(predata.shape[0],predata.shape[1])*255
    # val_pred = .transpose(0, 1) * 255
    # val_label = binary_label[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_out = np.zeros((val_pred.shape[0], val_pred.shape[1], 3), dtype=np.uint8)
    val_out[:, :, 0] = val_pred
    # val_out[:, :, 1] = val_label
    val_gt[val_out == 255] = 255
    # epsilon = 1e-5
    # pix_embedding = pix_embedding[i].data.cpu().numpy()
    # pix_vec = pix_embedding / (np.sum(pix_embedding, axis=0, keepdims=True) + epsilon) * 255
    # pix_vec = np.round(pix_vec).astype(np.uint8).transpose(1, 2, 0)
    # ins_label = instance_label[i].data.cpu().numpy().transpose(0, 1)
    # ins_label = np.repeat(np.expand_dims(ins_label, -1), 3, -1)
    # val_img = np.concatenate((val_gt, pix_vec, ins_label), axis=0)
    # val_img = np.concatenate((val_gt, pix_vec), axis=0)
    # return val_img
    return val_gt

if __name__ == '__main__':
    model_path = './checkpoints-combine-new1/83_checkpoint.pth'
    gpu = True
    if not torch.cuda.is_available():
        gpu = False

    # device = torch.device('cpu')

    model = LaneNet()
    model.to(DEVICE)
   # model = nn.DataParallel(model, device_ids=[0,7])

    # model.load_state_dict(torch.load(model_path, map_location=device))

    # if gpu:
    #     model = model.cuda()
    # print('loading pretrained model from %s' % model_path)
    if gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    model.eval()

    sourceFileName='ch0_20200318140335_20200318140435'
    video_path = os.path.join("/workspace/lanenet-lane-detection-11000", sourceFileName+'.mp4')
    times=0
    frameFrequency=1
    camera = cv2.VideoCapture(video_path)

    video=cv2.VideoWriter('./vedio/test.avi',cv2.VideoWriter_fourcc(*'MJPG'),25,(1280, 720))

    while True:
        res, imgori = camera.read()
        # cv2.imwrite('./vedio/pic/random%s.jpg'%(str(times)),imgori)
        times=times+1
        if not res:
            print('not res , not image')
            break
        if times%frameFrequency==0:
            # print('--------------------------------------------')
            # cv2.imwrite(outPutDirName + str(times)+'.jpg', image)
            transform=transforms.Compose([Rescale((1280, 720))])
            imgori=transform(imgori)
            # img=imgori
            #print(img.shape)
            toPIL = transforms.ToPILImage()
            img = np.asarray(toPIL(imgori[:,:,[2,1,0]]))
            # imgori = transforms.ColorJitter(brightness=0.0001)(imgori)
            # img0 = transform(np.asarray(imgori))
            # img=img0

            img=np.transpose(img,(2,0,1))

            #img = img.reshape(img.shape[2], img.shape[0], img.shape[1])

            img = np.expand_dims(img,0)
            # toTensor=transforms.ToTensor()
            # imgdata=Variable(img).type(torch.FloatTensor).cuda()
            print(img.shape)
            # imgTensor=toTensor(img)
            #img = np.expand_dims(img,0)
            imgdata=Variable(torch.from_numpy(img)).type(torch.FloatTensor).to(DEVICE)
            # imgdata=imgdata.unsqueeze(0)
            print(imgdata.size())
            output=model(imgdata)
            binary_seg_pred=output["binary_seg_pred"]

            # out=compose_img(imgdata,binary_seg_pred)

            binary_seg_pred = binary_seg_pred.squeeze(0)
            binary_seg_pred1=binary_seg_pred.to(torch.float32).cpu()
                # .numpy()
            # binary_seg_pred1=np.transpose(binary_seg_pred1,(1,2,0))
            # # print(binary_seg_pred1.shape)
            # binary_seg_pred1=binary_seg_pred1.reshape(binary_seg_pred1.shape[1],binary_seg_pred1.shape[2],binary_seg_pred1.shape[0])
            # # print(binary_seg_pred1[0])
            # # binary_seg_pred1 = binary_seg_pred.squeeze(0).cpu().numpy()
            # # pic.save('./vedio/pic/random%s.jpg'%(str(int( round(time_stamp * 1000) ))))
            pic=toPIL(binary_seg_pred1)
            imgx = cv2.cvtColor(np.asarray(pic),cv2.COLOR_RGB2BGR)
            imgx[np.where((imgx!=[0, 0, 0]).all(axis=2))] = [255,255,255]
            # # final_img[np.where((final_img==[255, 255, 255]).all(axis=2))] = [0,0,255]
            # #cv2.imwrite('./vedio/pic/random%s.jpg'%(str(times)),imgx)
            # time_stamp = time.time()
            #
            # # img2 = cv2.merge((imgx,imgx,imgx))
            #
            print (imgori.shape)
            print (imgx.shape)
            #
            src7 = cv2.addWeighted(imgori,0.8,imgx,1,0)
            #
            # #pic.save('./vedio/pic/random%s.jpg'%(str(int( round(time_stamp * 1000) ))))
            # # array1 = binary_seg_pred1.numpy()#to numpy array
            # # array1 = array1.reshape(array1.shape[1], array1.shape[2], array1.shape[0])
            # # final_img=np.uint8(array1)
            # # pic = toPIL()
            # # pic.save('./random1.jpg')
            # #print(binary_seg_pred1.shape)
            final_img=cv2.resize(src7,(1280, 720))
            final_img[np.where((final_img==[255, 255, 255]).all(axis=2))] = [0,0,255]
            # print(final_img)
            video.write(final_img)
            print("frame"+str(times)+str(times))
    print('-----------------------end---------------------------------')
    camera.release()
    video.release()

