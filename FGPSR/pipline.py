import torch
import numpy as np
import cv2
from my_deblocking_fliter import deblocking_h265, deblocking_weight_fusion
from time_cal import TimeRecorder
import os
import math
import model
from utils import load_pretrained_state_dict
import torch.nn as nn
import argparse
import yaml
from torchvision import utils as vutils
# from imgproc import tensor_to_image

from my_deblocking_fliter import deblocking_h265, deblocking_weight_fusion
import tqdm
import time
from deblocking_net.debloc_net import deblock_net

from MobileNetV3_deblock import MobileNetV3_deblock
from DeblockGhostnet import DeblockGhostNet

from unet_deblock import deblock_unet

from SR_models.RCAN import RCAN, load_rcan_network, tensor2img
from ultralytics import YOLO
from time_cal import TimeRecorder
from data_util import crop_cpu, combine

from ops.OmniSR import build_model_OmniSR


def build_model_deblock():
    # model = deblock_net().cuda()
    # model.load_state_dict(torch.load('./deblocking_net/deblock_net_460.pth')['net'])

    # model = MobileNetV3_deblock().cuda()
    # model.load_state_dict(torch.load('./MobileNetV3_deblock_316.pth')['net'])

    # model = DeblockGhostNet().cuda()
    # model.load_state_dict(torch.load('./DeblockGhostNet_874.pth')['net'])

    model = deblock_unet().cuda()
    model.load_state_dict(torch.load('./deblock_unet_1cen_195.pth')['net'])

    model.eval()
    return model


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


def build_model_rcan():
    model = RCAN(n_resblocks=20, n_feats=64, res_scale=1, n_colors=3, rgb_range=255, scale=4, reduction=16,
                 n_resgroups=10)

    load_rcan_network('./SR_models/rcan_checkpoints/RCAN_branch3.pth', model)

    model.cuda()
    model.eval()
    return model


def xy2index(x, y, img_width, block_size=16):
    """
    给定一个点的x,y坐标，返回该坐标所在的小块的下标。

    :param x: 点的x坐标
    :param y: 点的y坐标
    :param img: 大图,cv2
    :param block_size: 每个小块的尺寸
    :return: 小块的下标（index）
    """
    # 计算x，y坐标所在的行号和列号
    col = x // block_size
    row = y // block_size

    blocks_per_row = img_width // block_size

    # 计算下标
    index = row * blocks_per_row + col
    return index


def main():


    lr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\LRbicx4'
    hr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\GT'

    lr_name = os.listdir(lr_path)
    lr_name.sort(key=lambda x: int(x[:-4]))

    # model
    # g_model = build_model_rcan()
    # g_model = build_model_esrgan()
    g_model = build_model_OmniSR()

    psnr = 0.0
    num = 0

    TR = TimeRecorder(benchmark=False)

    for name in tqdm.tqdm(lr_name):
        num += 1

        lr = cv2.imread(os.path.join(lr_path, name), cv2.IMREAD_UNCHANGED)
        hr = cv2.imread(os.path.join(hr_path, name))



        # lr = lr.astype(np.float32)
        lr = lr.astype(np.float32) / 255.

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        lr_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))).float().unsqueeze(dim=0).cuda()
        # lr_tensor = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1).unsqueeze(dim=0).cuda()

        with torch.no_grad():
            TR.start()
            sr_tensor = g_model(lr_tensor)
            TR.end()

            # sr = tensor2img(sr_tensor.detach()[0].float().cpu(), out_type=np.uint8, min_max=(0, 255))  # uint8


            sr = sr_tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
            sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        sr = sr[:1024, :, :]
        hr = hr[:1024, :, :]

        psnr += cv2.PSNR(sr, hr)

        cv2.imwrite('./sr_imgs_3-16/OmiSR/OmiSR_crop/%s' % name, sr)

    print('psnr: %g, time: %g ms' % (psnr / num, TR.avg_time()))

    print('done')


if __name__ == "__main__":
    main()
