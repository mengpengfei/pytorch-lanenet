# coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lanenet.model.loss import DiscriminativeLoss
from lanenet.model.BiseNet_v2_2 import BiSeNet
from lanenet import config

import numpy as np

DEVICE = torch.device(config.gpu_no if torch.cuda.is_available() else 'cpu')
# from lanenet.model import lanenet_postprocess

class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        self.biSeNet=BiSeNet()
        self.no_of_instances=config.no_of_instances
        self.bn = nn.BatchNorm2d(128)
        self._pix_layer = nn.Conv2d(in_channels=128, out_channels=self.no_of_instances, kernel_size=1, bias=False).to(
            DEVICE)
        self.relu = nn.ReLU().to(DEVICE)

    def forward(self, input_tensor):

        biseNet_out=self.biSeNet(input_tensor)

        binary_seg_logits=biseNet_out["binary_seg_logits"]

        softmax_out=F.softmax(binary_seg_logits, dim=1)
        if config.is_training==2:
            tensor1=torch.zeros_like(softmax_out)
            tensor2=torch.ones_like(softmax_out)
            softmax_out=torch.where(softmax_out <=0.5 , tensor1, tensor2)
        binary_seg_ret = torch.argmax(softmax_out, dim=1, keepdim=True)

        pix_embedding = self.relu(self._pix_layer(self.bn(biseNet_out["instance_seg_logits"])))

        # instance_seg_logits=pix_embedding.squeeze(0)
        # print(instance_seg_logits.size())
        # instance_seg_logits=instance_seg_logits.permute(1, 2, 0)
        #
        # binary_seg_pred=binary_seg_ret.squeeze(0)
        # binary_seg_pred=binary_seg_pred.squeeze(0)
        if config.is_training==1:
            return {
                'instance_seg_logits': pix_embedding,
                'binary_seg_pred': binary_seg_ret,
                'binary_seg_logits': biseNet_out["binary_seg_logits"],
                'bsb_out1': biseNet_out["bsb_out1"],
                'bsb_out2': biseNet_out["bsb_out2"],
                'sg_out1': biseNet_out["sg_out1"],
                'sg_out3': biseNet_out["sg_out3"],
                'sg_out4': biseNet_out["sg_out4"],
                'sg_out5': biseNet_out["sg_out5"]
            }
        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': biseNet_out["binary_seg_logits"]
        }

if __name__ == '__main__':

    print(1==2)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    input = torch.rand(1, 3, 224, 1280)
    model = LaneNet()
    model.eval()
    print(model)
    output = model(input)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('BiSeNet_v2', output["instance_seg_logits"].size())
    print('BiSeNet_v2', output["binary_seg_pred"].size())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # postprocess_result = postprocessor.postprocess(
    #     binary_seg_result=binary_seg_image[0],
    #     instance_seg_result=instance_seg_image[0],
    #     source_image=image_vis
    # )
    # mask_image = postprocess_result['mask_image']
    #
    # for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
    #     instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
    # embedding_image = np.array(instance_seg_image[0], np.uint8)


def compute_loss(net_output, binary_label, instance_label,w1=0.25,w2=0.25,w3=0.25,w4=0.25):
    k_binary = 1.0
    k_instance = 0
    k_dist = 0

    # k_binary = 0.7
    # k_instance = 0.3
    # k_dist = 1.0

    ce_loss_fn = nn.CrossEntropyLoss()
    binary_seg_logits = net_output["binary_seg_logits"]
    bsb_out1 = net_output["bsb_out1"]
    bsb_out2 = net_output["bsb_out2"]

    sg_out1 = net_output["sg_out1"]
    sg_out3 = net_output["sg_out3"]
    sg_out4 = net_output["sg_out4"]
    sg_out5 = net_output["sg_out5"]
    # binary_seg_logits = net_output["binary_seg_pred"]
    binary_loss0 = ce_loss_fn(binary_seg_logits, binary_label)
    binary_loss1 = ce_loss_fn(bsb_out1, binary_label)
    binary_loss2 = ce_loss_fn(bsb_out2, binary_label)

    binary_loss3 = ce_loss_fn(sg_out1, binary_label)
    binary_loss4 = ce_loss_fn(sg_out3, binary_label)
    binary_loss5 = ce_loss_fn(sg_out4, binary_label)
    binary_loss6 = ce_loss_fn(sg_out5, binary_label)

    binary_loss=w1*binary_loss0+w2*binary_loss1+w3*binary_loss2+w4*(binary_loss3+binary_loss4+binary_loss5+binary_loss6)/4
    # binary_loss=(binary_loss0+binary_loss1+binary_loss2+(binary_loss3+binary_loss4+binary_loss5+binary_loss6)/4)/4
    # binary_loss=(binary_loss0+binary_loss1+binary_loss2)/3

    pix_embedding = net_output["instance_seg_logits"]
    # ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    ds_loss_fn = DiscriminativeLoss(0.4, 3.0, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    instance_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    total_loss = binary_loss + instance_loss + dist_loss
    out = net_output["binary_seg_pred"]
    iou = 0
    batch_size = out.size()[0]
    k=0
    for i in range(batch_size):
        PR = out[i].squeeze(0).nonzero().size()[0]
        GT = binary_label[i].nonzero().size()[0]
        TP = (out[i].squeeze(0) * binary_label[i]).nonzero().size()[0]
        union = PR + GT - TP
        if union!=0:
            iou += TP / union
            k+=1
    iou = iou / k
    return total_loss, binary_loss, instance_loss, out, iou
