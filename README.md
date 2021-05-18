# pytorch-lanenet
效果很好的lanenet网络，主干网络基于bisenetv2并对主干网络做了修改，效果远好于bisnetv2
可直接训练自己的数据应用于生产

inspired by https://github.com/MaybeShewill-CV/lanenet-lane-detection

Using Bisenetv2 as Encoder.

使用步骤：

1、安装pytorch环境，pytorch官网有说明，推荐使用docker

2、生成样本的train.txt和val.txt文件

文件内容：

原始图片 语义分割图 实例分割图

3、修改 script目录下的Convert2LMDB.py main函数中的txt文件路径和生成的lmdb文件名

4、修改lanenet/train.py中的train_dataset_file和val_dataset_file为自己生成的lmdb文件路径

按照自己的需要lanenet/config.py文件中的配置参数

5、执行 python setup.py install

6、执行python lanenet/train.py --lr 0.001 --val True --bs 16 --save ./checkpoints --w1 0.25 --w2 0.25 --w3 0.25 --w4 0.25 --epochs 200 

开始训练。

7、训练完成后 修改 lanenet/online_test_video.py 中的模型路径和视频路径，测试视频‘



TODO：
1、将代码封装，抽取配置，整理代码

