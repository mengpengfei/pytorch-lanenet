# coding: utf-8
# number of unique pixel values
n_labels=6
#embeding特征数
no_of_instances=6
#分割目标的种类数 只分割车道线值为2
num_classes=2
#device_ids 中第一个gpu号
gpu_no='cuda:0'
# 使用的gpu
device_ids = [0, 1,2,3,4,5,6]
lr=0.001
epochs=2000
bs=16
show_interval=30
save_interval=3
# 1-train，2-val
is_training=1
# 模型保存位置
save_path='./checkpoints'
#loss权重设置详见compute_loss
w1=0.25
#loss权重设置详见compute_loss
w2=0.25
#loss权重设置详见compute_loss
w3=0.25
#loss权重设置详见compute_loss
w4=0.25
train_dataset_file = '/workspace/all/index/train1'
val_dataset_file = '/workspace/all/index/val1'