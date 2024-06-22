#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OmniSR.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:06:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import  torch
import  torch.nn as nn
from ops.OSAG import OSAG
from ops.pixelshuffle import pixelshuffle_block
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from ops.Recon_Net import deep_rec_dq,deep_rec
from ops.INV import InvRescaleNet,INV
import torch.nn.functional as F
from typing import Any


def build_model_OmniSR():

    kwards = {'upsampling': 4,
              'res_num': 5,
              'block_num': 1,
              'bias': True,
              'block_script_name': 'OSA',
              'block_class_name': 'OSA_Block',
              'window_size': 8,
              'pe': True,
              'ffn_bias': True}
    kwards_lr = {'channel_in' : 3,
                 'channel_out': 3,
                 'block_num': [8, 8],
                 'down_num': 2,
                 'down_scale': 4
    }

    g_model=OmniSR(kwards=kwards)

    g_model = g_model.to('cuda:0')

    g_model.load_state_dict(torch.load('./ops/Omni4_194.pt'))

    g_model.eval()

    return g_model

class DownSample_x2(nn.Module):
    def __init__(self):
        super(DownSample_x2, self).__init__()
        # self.layer=nn.Sequential(
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0),
        # nn.ReLU(),
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.down_x2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        # )

    def forward(self, x):
        x = self.conv1(x)

        redual = self.pool2d(x)
        out = self.down_x2_1(x)
        x = redual + out

        x = self.conv2(x)
        return x

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        # self.layer=nn.Sequential(
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0),
        # nn.ReLU(),
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.down_x2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.down_x2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        # )

    def forward(self, x):
        x = self.conv1(x)

        redual = self.pool2d(x)
        out = self.down_x2_1(x)
        x = redual + out

        redual = self.pool2d(x)
        out = self.down_x2_2(x)
        x = redual + out
        # x = out

        x = self.conv2(x)
        return x

class DownSample_x8(nn.Module):
    def __init__(self):
        super(DownSample_x8, self).__init__()
        # self.layer=nn.Sequential(
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0),
        # nn.ReLU(),
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.down_x2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.down_x2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        # self.prelu = nn.PReLU()
        self.down_x2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        # )

    def forward(self, x):
        x = self.conv1(x)

        redual = self.pool2d(x)
        out = self.down_x2_1(x)
        x = redual + out

        redual = self.pool2d(x)
        out = self.down_x2_2(x)
        x = redual + out

        redual = self.pool2d(x)
        out = self.down_x2_3(x)
        x = redual + out

        x = self.conv2(x)
        return x


class OmniSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super(OmniSR, self).__init__()
        # print(kwargs)
        kwargs = kwargs['kwards']
        res_num = kwargs["res_num"]
        up_scale = kwargs["upsampling"]
        bias = kwargs["bias"]

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)

        # self.osag1 = OSAG(channel_num=num_feat, **kwargs)
        # self.osag2 = OSAG(channel_num=num_feat, **kwargs)
        # self.osag3 = OSAG(channel_num=num_feat, **kwargs)
        # self.osag4 = OSAG(channel_num=num_feat, **kwargs)
        # self.osag5 = OSAG(channel_num=num_feat, **kwargs)

        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                               bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                                bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        self.window_size = kwargs["window_size"]
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        # print(x.shape)
        H, W = x.shape[2:]
        # print('aa',x.shape)
        x = self.check_image_size(x)

        residual = self.input(x)
        # print(x.shape)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        return out

class LR_SR_x2(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x2, self).__init__()
        self.layer1 = DownSample_x2()
        self.layer2 = OmniSR(kwards=kwards)

    def forward(self, x):
        # x.requires_grad_(True)
        # LR = checkpoint(self.layer1, x, use_reentrant=True)
        LR = self.layer1(x)
        # LR.requires_grad_(True)
        # HR = checkpoint(self.layer2, LR, use_reentrant=True)
        HR = self.layer2(LR)
        return LR,HR

class LR_SR(nn.Module):
    def __init__(self, kwards):
        super(LR_SR, self).__init__()
        self.layer1 = DownSample()
        self.layer2 = OmniSR(kwards=kwards)

    def forward(self, x):
        # x.requires_grad_(True)
        # LR = checkpoint(self.layer1, x, use_reentrant=True)
        LR = self.layer1(x)
        # LR.requires_grad_(True)
        # HR = checkpoint(self.layer2, LR, use_reentrant=True)
        HR = self.layer2(LR)
        return LR,HR
class LR_SR_x8(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x8, self).__init__()
        self.layer1 = DownSample_x8()
        self.layer2 = OmniSR(kwards=kwards)

    def forward(self, x):
        # x.requires_grad_(True)
        # LR = checkpoint(self.layer1, x, use_reentrant=True)
        LR = self.layer1(x)
        # LR.requires_grad_(True)
        # HR = checkpoint(self.layer2, LR, use_reentrant=True)
        HR = self.layer2(LR)
        return LR,HR

class LR_SR_Rec_x4(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_Rec_x4, self).__init__()
        self.layer1 = DownSample()
        self.layer2 = OmniSR(kwards=kwards)
        self.layer3 = deep_rec_dq()

    def forward(self, x):
        LR = self.layer1(x)
        HR = self.layer2(LR)
        HR2 = self.layer3(HR,LR,self.layer1,self.layer2)
        return LR,HR,HR2

class LR_SR_INV(nn.Module):
    def __init__(self, kward_lr,kwards_sr):
        super(LR_SR_INV, self).__init__()
        self.layer1 = INV(**kward_lr)
        self.layer2 = OmniSR(kwards=kwards_sr)
        self.split_len1 = 3
        self.split_len2 = 48 - 3

    def forward(self, x, rev=False, cal_jacobian=False):
        LR = self.layer1(x, rev=rev, cal_jacobian=cal_jacobian)
        LR, LR_feature = (LR.narrow(1, 0, self.split_len1), LR.narrow(1, self.split_len1, self.split_len2))
        rand_noise = torch.randn(LR.shape[0], LR_feature.shape[1], LR.shape[2], LR.shape[3]).cuda()
        rand_noise = torch.cat((LR, rand_noise), 1)
        # HR_temp=checkpoint(self.layer1,rand_noise,True,use_reentrant=True)
        HR_temp=self.layer1(rand_noise,rev=True)
        HR = self.layer2(LR)
        return LR, LR_feature, HR_temp, HR