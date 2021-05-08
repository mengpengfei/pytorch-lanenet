import time
import os
import sys

from tqdm import tqdm

import torch
from lanenet.dataloader.lmdb_data_loaders import LaneDataSet
from lanenet.dataloader.transformers import Rescale
from lanenet.model.model import LaneNet, compute_loss
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from torchvision import transforms

from lanenet.utils.cli_helper import parse_args
from lanenet.utils.average_meter import AverageMeter
from lanenet.test import test

import numpy as np
import cv2
from lanenet import config

# might want this in the transformer part as well
# VGG_MEAN = [103.939, 116.779, 123.68]

DEVICE = torch.device(config.gpu_no if torch.cuda.is_available() else 'cpu')


def compose_img(image_data, out, binary_label, pix_embedding, instance_label, i):
    val_gt = (image_data[i].cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
    val_pred = out[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_label = binary_label[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_out = np.zeros((val_pred.shape[0], val_pred.shape[1], 3), dtype=np.uint8)
    val_out[:, :, 0] = val_pred
    val_out[:, :, 1] = val_label
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

def train(train_loader, model, optimizer, epoch,w1,w2,w3,w4):
    model.train()
    batch_time = AverageMeter()
    mean_iou = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    end = time.time()
    step = 0

    t = tqdm(enumerate(iter(train_loader)), leave=False, total=len(train_loader))

    for batch_idx, batch in t:
        try:
            step += 1
            image_data = Variable(batch[0]).type(torch.FloatTensor).to(DEVICE)
            binary_label = Variable(batch[1]).type(torch.LongTensor).to(DEVICE)
            instance_label = Variable(batch[2]).type(torch.FloatTensor).to(DEVICE)

            #print("///////////////////////////////////////////////////")
            #print(image_data.size())
            #print(binary_label.size()) 
            # # print(image_data.shape)
            #print("///////////////////////////////////////////////////")

            # forward pass
            net_output = model(image_data)

            # compute loss
            total_loss, binary_loss, instance_loss, out, train_iou = compute_loss(net_output, binary_label, instance_label,w1,w2,w3,w4)

            # update loss in AverageMeter instance
            total_losses.update(total_loss.item(), image_data.size()[0])
            binary_losses.update(binary_loss.item(), image_data.size()[0])
            instance_losses.update(instance_loss.item(), image_data.size()[0])
            mean_iou.update(train_iou, image_data.size()[0])

            # reset gradients
            optimizer.zero_grad()

            # backpropagate
            total_loss.backward()

            # update weights
            optimizer.step()

            # update batch time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % config.show_interval == 0:
                print(
                    "Epoch {ep} Step {st} |({batch}/{size})| ETA: {et:.2f}|Total loss:{tot:.5f}|Binary loss:{bin:.5f}|Instance loss:{ins:.5f}|IoU:{iou:.5f}".format(
                        ep=epoch + 1,
                        st=step,
                        batch=batch_idx + 1,
                        size=len(train_loader),
                        et=batch_time.val,
                        tot=total_losses.avg,
                        bin=binary_losses.avg,
                        ins=instance_losses.avg,
                        iou=train_iou,
                    ))
                print("current learning rate is %s"%(str(optimizer.state_dict()['param_groups'][0]['lr'])))
                sys.stdout.flush()
                train_img_list = []
                for i in range(3):
                    train_img_list.append(
                        compose_img(image_data, out, binary_label, net_output["instance_seg_logits"], instance_label, i))
                train_img = np.concatenate(train_img_list, axis=1)
                cv2.imwrite(os.path.join("./output", "train_" + str(epoch + 1) + "_step_" + str(step) + ".png"), train_img)
        except Exception as e:
            print(e)
            print('error')
    return mean_iou.avg


def save_model(save_path, epoch, model):
    save_name = os.path.join(save_path, f'{epoch}_checkpoint.pth')
    torch.save(model.module.state_dict(), save_name)
    print("model is saved: {}".format(save_name))


def main():
    # args = parse_args()

    save_path = config.save_path
    w1 = config.w1
    w2 = config.w2
    w3 = config.w3
    w4 = config.w4

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = config.train_dataset_file
    val_dataset_file = config.val_dataset_file

    train_dataset = LaneDataSet(train_dataset_file, transform=None)
    # train_dataset = LaneDataSet(train_dataset_file, transform=transforms.Compose([Rescale((1280, 720))]))
    train_loader = DataLoader(train_dataset, batch_size=config.bs, shuffle=True,num_workers=4,pin_memory=True,drop_last=True)

    # if args.val:
    val_dataset = LaneDataSet(val_dataset_file, transform=None)
    # val_dataset = LaneDataSet(val_dataset_file, transform=transforms.Compose([Rescale((1280, 720))]))
    val_loader = DataLoader(val_dataset, batch_size=config.bs, shuffle=True,num_workers=4,pin_memory=True,drop_last=True)

    model = LaneNet()
    model = nn.DataParallel(model, device_ids=config.device_ids)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print(f"{config.epochs} epochs {len(train_dataset)} training samples\n")
    log_model="/workspace/pytorch-lanenet-master/checkpoints-combine-new1/83_checkpoint_state.pth"
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(log_model):
        checkpoint = torch.load(log_model)
        model.module.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        #for p in optimizer.param_groups:
        #     p['lr'] = args.lr

        start_epoch = int(checkpoint['epoch'])+1
        # start_epoch = 272
        print('load epoch {} success'.format(start_epoch-1))
    else:
        start_epoch = 0
        print('no model,will start train from 0 epoche')

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch {epoch}")
        train_iou = train(train_loader, model, optimizer, epoch,w1,w2,w3,w4)
        # if args.val:
        val_iou = test(val_loader, model, epoch)
        if (epoch+1) % config.save_interval == 0:
            save_model(save_path, epoch, model)
            save_state_name = os.path.join(save_path, f'{epoch}_checkpoint_state.pth')
            checkpoint = {
                "net": model.module.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, save_state_name)
        print(f"Train IoU : {train_iou}")
        # if args.val:
        print(f"Val IoU : {val_iou}")


if __name__ == '__main__':
    main()
