import cv2
import os
import time
#要提取视频的文件名，隐藏后缀
sourceFileName='ch0_20200318140335_20200318140435'
#在这里把后缀接上
video_path = os.path.join("/workspace/lanenet-lane-detection-11000", sourceFileName+'.mp4')
times=0
#提取视频的频率，每１帧提取一个
frameFrequency=1
print(video_path)
camera = cv2.VideoCapture(video_path)

file_dir='./vedio/ch0_20200318140335_20200318140435/'
video=cv2.VideoWriter('./vedio/test.avi',cv2.VideoWriter_fourcc(*'MJPG'),25,(1280,720))  #定义保存视频目录名称及压缩格式，fps=10,像素为1280*720

while True:
    times=times+1
    res, image = camera.read()
    print(res)
    print(image.shape)
    if not res:
        print('not res , not image')
        break
    if times%frameFrequency==0:
        # cv2.imwrite(outPutDirName + str(times)+'.jpg', image)
        image=cv2.resize(image,(1280,720)) #将图片转换为1280*720
        video.write(image)   #写入视频
print('-----------------------end---------------------------------')
camera.release()
video.release()