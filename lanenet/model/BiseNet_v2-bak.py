import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models
from lanenet import config
import collections

class conv2d(nn.Module):
    def __init__(self,in_dim,out_dim,k,pad,stride,groups = 1,bias=False,use_bn = True,use_rl = True):
        super(conv2d,self).__init__()
        self.use_bn = use_bn
        self.use_rl = use_rl
        self.conv = nn.Conv2d(in_dim,out_dim,k,padding=pad,stride=stride, groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,bottom):
        if self.use_bn and self.use_rl:
            return self.relu(self.bn(self.conv(bottom)))
        elif self.use_bn:
            return self.bn(self.conv(bottom))
        else:
            return self.conv(bottom)

class SegHead(nn.Module):
    def __init__(self,in_dim,out_dim,cls,size=[720,1280]):
        super(SegHead,self).__init__()
        self.size = size
        self.conv = conv2d(in_dim,out_dim,3,1,1)
        self.cls = conv2d(out_dim,cls,1,0,1,use_bn=False,use_rl=False)
    def forward(self,feat):
        x = self.conv(feat)
        x = self.cls(x)
        pred = F.interpolate(x, size=self.size, mode="bilinear",align_corners=True)
        return pred

class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock,self).__init__()
        self.conv1 = conv2d(3,16,3,1,2)
        self.conv_1x1 = conv2d(16,8,1,0,1)
        self.conv_3x3 = conv2d(8,16,3,1,2)
        self.mpooling = nn.MaxPool2d(3,2,1)
        self.conv2 = conv2d(32,16,3,1,1)
    def forward(self,bottom):
        base = self.conv1(bottom)
        conv_1 = self.conv_1x1(base)
        conv_3 = self.conv_3x3(conv_1)
        pool = self.mpooling(base)
        cat = torch.cat([conv_3,pool],1)
        res = self.conv2(cat)
        return res

class ContextEmbeddingBlock(nn.Module):
    def __init__(self,in_dim):
        super(ContextEmbeddingBlock,self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)#1
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = conv2d(in_dim,in_dim,1,0,1)
        self.conv2 = conv2d(in_dim,in_dim,3,1,1,use_bn = False,use_rl = False)
    def forward(self,bottom):
        gap = self.gap(bottom)
        bn = self.bn1(gap)
        conv1 = self.conv1(bn)
        feat = bottom+conv1
        res = self.conv2(feat)
        return res

class GatherExpansion(nn.Module):
    def __init__(self,in_dim,out_dim,stride = 1,exp = 6):
        super(GatherExpansion,self).__init__()
        exp_dim = in_dim*exp
        self.stride = stride
        self.conv1 = conv2d(in_dim,exp_dim,3,1,1)
        self.dwconv2 = conv2d(exp_dim,exp_dim,3,1,1,exp_dim,use_rl = False)
        self.conv_11 = conv2d(exp_dim,out_dim,1,0,1,use_rl = False)

        self.dwconv1 = conv2d(exp_dim,exp_dim,3,1,2,exp_dim,use_rl = False)
        self.dwconv3 = conv2d(in_dim,in_dim,3,1,2,in_dim,use_rl = False)
        self.conv_12 = conv2d(in_dim,out_dim,1,0,1,use_rl = False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,bottom):
        base = self.conv1(bottom)
        if self.stride == 2:
            base = self.dwconv1(base)
            bottom = self.dwconv3(bottom)
            bottom = self.conv_12(bottom)
        x = self.dwconv2(base)
        x = self.conv_11(x)
        res = self.relu(x+bottom)
        return res

class BGA(nn.Module):
    def __init__(self,in_dim):
        super(BGA,self).__init__()
        self.in_dim = in_dim
        self.db_dwconv = conv2d(in_dim,in_dim,3,1,1,in_dim,use_rl=False)
        self.db_conv1x1 = conv2d(in_dim,in_dim,1,0,1,use_rl=False,use_bn=False)
        self.db_conv = conv2d(in_dim,in_dim,3,1,2,use_rl=False)
        self.db_apooling = nn.AvgPool2d(3,2,1)

        self.sb_dwconv = conv2d(in_dim,in_dim,3,1,1,in_dim,use_rl=False)
        self.sb_conv1x1 = conv2d(in_dim,in_dim,1,0,1,use_rl=False,use_bn=False)
        self.sb_conv = conv2d(in_dim,in_dim,3,1,1,use_rl=False)
        self.sb_sigmoid = nn.Sigmoid()

        self.conv = conv2d(in_dim,in_dim,3,1,1,use_rl=False)
    def forward(self,db,sb):
        db_dwc = self.db_dwconv(db)
        db_out = self.db_conv1x1(db_dwc)#
        db_conv = self.db_conv(db)
        db_pool = self.db_apooling(db_conv)

        sb_dwc = self.sb_dwconv(sb)
        sb_out = self.sb_sigmoid(self.sb_conv1x1(sb_dwc))#
        sb_conv = self.sb_conv(sb)
        sb_up = self.sb_sigmoid(F.interpolate(sb_conv, size=db_out.size()[2:], mode="bilinear",align_corners=True))
        db_l = db_out*sb_up
        sb_r = F.interpolate(sb_out*db_pool, size=db_out.size()[2:], mode="bilinear",align_corners=True)
        res = self.conv(db_l+sb_r)
        return res

class DetailedBranch(nn.Module):
    def __init__(self):
        super(DetailedBranch,self).__init__()
        self.s1_conv1 = conv2d(3,64,3,1,2)
        self.s1_conv2 = conv2d(64,64,3,1,1)

        self.s2_conv1 = conv2d(64,64,3,1,2)
        self.s2_conv2 = conv2d(64,64,3,1,1)
        self.s2_conv3 = conv2d(64,64,3,1,1)

        self.s3_conv1 = conv2d(64,128,3,1,2)
        self.s3_conv2 = conv2d(128,128,3,1,1)
        self.s3_conv3 = conv2d(128,128,3,1,1)
    def forward(self,bottom):

        detail_stage_outputs = collections.OrderedDict()

        s1_1 = self.s1_conv1(bottom)
        s1_2 = self.s1_conv2(s1_1)

        detail_stage_outputs["stg1"] = s1_2

        s2_1 = self.s2_conv1(s1_2)
        s2_2 = self.s2_conv2(s2_1)
        s2_3 = self.s2_conv3(s2_2)

        detail_stage_outputs["stg2"] = s2_3

        s3_1 = self.s3_conv1(s2_3)
        s3_2 = self.s3_conv2(s3_1)
        s3_3 = self.s3_conv3(s3_2)

        detail_stage_outputs["stg3"] = s3_3

        return {
            'out': s3_3,
            'detail_stage_outputs': detail_stage_outputs
        }

class SemanticBranch(nn.Module):
    def __init__(self):
        super(SemanticBranch,self).__init__()
        self.stem = StemBlock()
        self.s3_ge1 = GatherExpansion(16,32,2)
        self.s3_ge2 = GatherExpansion(32,32)

        self.s4_ge1 = GatherExpansion(32,64,2)
        self.s4_ge2 = GatherExpansion(64,64)

        self.s5_ge1 = GatherExpansion(64,128,2)
        self.s5_ge2 = GatherExpansion(128,128)
        self.s5_ge3 = GatherExpansion(128,128)
        self.s5_ge4 = GatherExpansion(128,128)
        self.s5_ge5 = GatherExpansion(128,128,exp=1)

        self.ceb = ContextEmbeddingBlock(128)

    def forward(self,bottom):
        seg_stage_outputs = collections.OrderedDict()

        stg1 = self.stem(bottom)
        #print(stg12.size())
        seg_stage_outputs["stg1"] = stg1

        stg3 = self.s3_ge1(stg1)
        stg3 = self.s3_ge2(stg3)
        #print(stg3.size())
        seg_stage_outputs["stg3"] = stg3

        stg4 = self.s4_ge1(stg3)
        stg4 = self.s4_ge2(stg4)

        seg_stage_outputs["stg4"] = stg4
        #print(stg4.size())
        stg5 = self.s5_ge1(stg4)
        stg5 = self.s5_ge2(stg5)
        stg5 = self.s5_ge3(stg5)
        stg5 = self.s5_ge4(stg5)
        stg5 = self.s5_ge5(stg5)

        seg_stage_outputs["stg5"] = stg5
        #print(stg5.size())
        out = self.ceb(stg5)

        return {
            'out': out,
            'seg_stage_outputs': seg_stage_outputs
        }

class InstanceSegmentationBranch(nn.Module):
    def __init__(self):
        super(InstanceSegmentationBranch,self).__init__()
        self.bsconv1 = conv2d(128,256,3,1,1,use_rl=True)
        self.bsconv2 = conv2d(256,128,1,0,1,use_rl=True)
    def forward(self,data):
        input_tensor_size=list(data.size())
        tmp_size=input_tensor_size[2:]
        out_put_tensor_size=tuple([int(tmp * 8) for tmp in tmp_size])
        conv1_out=self.bsconv1(data)
        conv2_out=self.bsconv2(conv1_out)
        isb_out = F.interpolate(conv2_out, size=out_put_tensor_size, mode="bilinear",align_corners=True)
        return isb_out

class BinarySegmentationBranch(nn.Module):

    def __init__(self):
        super(BinarySegmentationBranch,self).__init__()

        self.bsconv1_pre = conv2d(128,32,3,1,1,use_rl=True)
        self.bsconv1_pre1 = conv2d(32,64,3,1,1,use_rl=True)
        self.bsconv1_pre2 = conv2d(144,64,3,1,1,use_rl=True)
        self.bsconv1_pre3 = conv2d(64,128,3,1,1,use_rl=True)
        self.bsconv1_pre4 = conv2d(192,64,3,1,1,use_rl=True)
        self.bsconv1_pre5 = conv2d(64,128,3,1,1,use_rl=True)
        self.bsconv3 = conv2d(128,config.num_classes,1,0,1,use_rl=False,use_bn=True)


        # self.bsconv1 = conv2d(128,256,3,1,1,use_rl=True)
        # self.bsconv2 = conv2d(256,128,1,0,1,use_rl=True)
        # self.bsconv3 = conv2d(128,config.num_classes,1,0,1,use_rl=False,use_bn=True)

    def forward(self,data,seg_stage_outputs,detail_stage_outputs):

        input_tensor_size=list(data.size())
        tmp_size=input_tensor_size[2:]
        output_stage2_size=[int(tmp * 2) for tmp in tmp_size]
        output_stage1_size=[int(tmp * 4) for tmp in tmp_size]
        out_put_tensor_size=[int(tmp * 8) for tmp in tmp_size]

        out1=self.bsconv1_pre(data)
        out2=self.bsconv1_pre1(out1)
        output_stage2_tensor = F.interpolate(out2, size=tuple(output_stage2_size), mode="bilinear",align_corners=True)
        # output_stage2_tensor = tf.concat([output_stage2_tensor, detail_stage_outputs['stage_2'], semantic_stage_outputs['stage_1']], axis=-1, name='stage_2_concate_features')
        output_stage2_tensor=torch.cat([output_stage2_tensor,detail_stage_outputs['stg2'], seg_stage_outputs['stg1']],1)
        output_stage2_tensor=self.bsconv1_pre2(output_stage2_tensor)
        output_stage2_tensor=self.bsconv1_pre3(output_stage2_tensor)
        output_stage1_tensor = F.interpolate(output_stage2_tensor, size=tuple(output_stage1_size), mode="bilinear",align_corners=True)
        #output_stage1_tensor = tf.concat([output_stage1_tensor, detail_stage_outputs['stage_1']], axis=-1, name='stage_1_concate_features')
        output_stage1_tensor=torch.cat([output_stage1_tensor,detail_stage_outputs['stg1']],1)
        output_stage1_tensor=self.bsconv1_pre4(output_stage1_tensor)
        output_stage1_tensor=self.bsconv1_pre5(output_stage1_tensor)
        output_stage1_tensor=self.bsconv3(output_stage1_tensor)

        # conv_out1=self.bsconv1(data)
        # conv_out2=self.bsconv2(conv_out1)
        # conv_out3=self.bsconv3(conv_out2)

        # print("--------------------conv_out3 size--------------------")
        # print(list(conv_out3.size()))
        # print("--------------------bga size--------------------")

        # print("--------------------out_put_tensor_size--------------------")
        # print(out_put_tensor_size)
        # print(tuple(out_put_tensor_size))
        # print("--------------------out_put_tensor_size--------------------")
        bsb_out = F.interpolate(output_stage1_tensor, size=tuple(out_put_tensor_size), mode="bilinear",align_corners=True)
        return bsb_out

class BiSeNet(nn.Module):
    def __init__(self):
        super(BiSeNet, self).__init__()
        self.db = DetailedBranch()
        self.sb = SemanticBranch()
        self.bga = BGA(128)
        self._init_params()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.binarySegmentationBranch=BinarySegmentationBranch()
        self.instanceSegmentationBranch=InstanceSegmentationBranch()
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self,data,y=None):
        db = self.db(data)
        sb = self.sb(data)
        bga = self.bga(db["out"],sb["out"])
        # print("--------------------bga size--------------------")
        # print(bga.size())
        # print("--------------------bga size--------------------")
        bsb_res=self.binarySegmentationBranch(bga,sb["seg_stage_outputs"],db["detail_stage_outputs"])
        isb_res=self.instanceSegmentationBranch(bga)
        return {
            'instance_seg_logits': isb_res,
            # 'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': bsb_res
        }

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    input = torch.rand(1, 3, 256, 512).cuda()
    model = BiSeNet().cuda()
    model.eval()
    print(model)
    output = model(input)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('BiSeNet_v2', output["instance_seg_logits"].size())
    print('BiSeNet_v2', output["binary_seg_logits"].size())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
