'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch.nn as nn
from torch.nn import init

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthw

    ise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class MobileNetV3_deblock(nn.Module):
    def __init__(self, act=nn.Hardswish):
        super(MobileNetV3_deblock, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.enc = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 1),
            Block(5, 40, 120, 40, act, True, 1),
            # Block(5, 40, 120, 48, act, True, 1),
        )


        self.dec = nn.Sequential(
            # Block(5, 48, 120, 40, act, True, 1),
            Block(5, 40, 120, 40, act, True, 1),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Block(5, 40, 96, 24, act, True, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(3, 24, 64, 16, nn.ReLU, False, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Block(3, 16, 16, 16, nn.ReLU, True, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.hs2 = act(inplace=True)



    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.enc(out)
        out = self.dec(out)
        out = self.hs2(self.bn2(self.conv2(out)))

        return out

import yaml
from utils import load_pretrained_state_dict
import model


def build_model_esrgan():
    with open("./configs/test/ESRGAN_x4-DFO2K-Set5.yaml", "r") as f:
        config = yaml.full_load(f)

    g_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           growth_channels=config["MODEL"]["G"]["GROWTH_CHANNELS"],
                                                           num_rrdb=config["MODEL"]["G"]["NUM_RRDB"])

    g_model = g_model.to('cuda:0')
    g_model = load_pretrained_state_dict(g_model, False,
                                         "./results/pretrained_models/ESRGAN_x4-DFO2K.pth.tar")
    g_model.eval()

    return g_model


from time_cal import TimeRecorder
import torch
from torchvision import models
from unet_deblock import deblock_unet
import time
import numpy as np

if __name__=='__main__':
    TR = TimeRecorder(benchmark=False)
    TR_yure = TimeRecorder(benchmark=False)

    # model = MobileNetV3_denoise().cuda()

    # model = models.alexnet().cuda()

    # model = deblock_unet().cuda()

    model = build_model_esrgan()

    model.eval()
    # print(model)

    with torch.no_grad():
        input1 = torch.randn(1, 3, 16, 16).to('cuda:0')

        # 预热
        for i in range(5):
            TR_yure.start()
            y = model(input1)
            TR_yure.end()

        # input = np.ascontiguousarray(np.random.rand(480, 16, 16, 3).astype(np.float32).transpose(0, 3, 1, 2))
        # input = torch.from_numpy(input).to('cuda:0')

        # input = torch.randn(1, 3, 270, 480).to('cuda:0')
        TR.count()
        for i in range(5):
            TR.start()
            y = model(input)
            TR.end_t()
            # print(1)
    print(TR.avg_time())
    print(y)