import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models

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

class SegHead(nn.Module):
    def __init__(self,in_dim,out_dim,cls,size=[512,1024]):
        super(SegHead,self).__init__()
        self.size = size
        self.conv = conv2d(in_dim,out_dim,3,1,1)
        self.cls = conv2d(out_dim,cls,1,0,1,use_bn=False,use_rl=False)
    def forward(self,feat):
        x = self.conv(feat)
        x = self.cls(x)
        pred = F.interpolate(x, size=self.size, mode="bilinear",align_corners=True)
        return pred


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
        s1_1 = self.s1_conv1(bottom)
        s1_2 = self.s1_conv2(s1_1)

        s2_1 = self.s2_conv1(s1_2)
        s2_2 = self.s2_conv2(s2_1)
        s2_3 = self.s2_conv3(s2_2)

        s3_1 = self.s3_conv1(s2_3)
        s3_2 = self.s3_conv2(s3_1)
        s3_3 = self.s3_conv3(s3_2)
        return s3_3

class SemanticBranch(nn.Module):
    def __init__(self, cls):
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
        if self.training:
            self.seghead1 = SegHead(16,16,cls)
            self.seghead2 = SegHead(32,32,cls)
            self.seghead3 = SegHead(64,64,cls)
            self.seghead4 = SegHead(128,128,cls)

        self.ceb = ContextEmbeddingBlock(128)

    def forward(self,bottom):
        stg12 = self.stem(bottom)
        #print(stg12.size())
        stg3 = self.s3_ge1(stg12)
        stg3 = self.s3_ge2(stg3)
        #print(stg3.size())
        stg4 = self.s4_ge1(stg3)
        stg4 = self.s4_ge2(stg4)
        #print(stg4.size())
        stg5 = self.s5_ge1(stg4)
        stg5 = self.s5_ge2(stg5)
        stg5 = self.s5_ge3(stg5)
        stg5 = self.s5_ge4(stg5)
        stg5 = self.s5_ge5(stg5)
        #print(stg5.size())
        out = self.ceb(stg5)
        if self.training:
            seghead1 = self.seghead1(stg12)
            seghead2 = self.seghead2(stg3)
            seghead3 = self.seghead3(stg4)
            seghead4 = self.seghead4(stg5)
            return out,seghead1,seghead2,seghead3,seghead4
        else:
            return out


class BiSeNet(nn.Module):
    def __init__(self,cls):
        super(BiSeNet, self).__init__()
        self.db = DetailedBranch()
        self.sb = SemanticBranch(cls)
        self.bga = BGA(128)
        self.seghead = SegHead(128,128,cls)
        self._init_params()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
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
        if self.training:
            sb,head1,head2,head3,head4 = self.sb(data)
        else:
            sb = self.sb(data)
        bga = self.bga(db,sb)
        pred = self.seghead(bga)
        if self.training:
            main_loss = self.criterion(pred, y)
            aux1_loss = self.criterion(head1, y)
            aux2_loss = self.criterion(head2, y)
            aux3_loss = self.criterion(head3, y)
            aux4_loss = self.criterion(head4, y)
            return pred.max(1)[1],main_loss,(aux1_loss,aux2_loss,aux3_loss,aux4_loss)
        return pred

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.rand(4, 3, 720, 960).cuda()
    model = BiSeNet(11,False).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('BiSeNet', output.size())