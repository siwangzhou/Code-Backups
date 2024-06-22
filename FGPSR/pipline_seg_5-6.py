from skimage.metrics import structural_similarity as compare_ssim
import torch
import numpy as np
import cv2
from my_deblocking_fliter import deblocking_h265, deblocking_weight_fusion
from time_cal import TimeRecorder
import os
import time
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
from data_util import imresize_np
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
    model = RCAN(n_resblocks=20, n_feats=36, res_scale=1, n_colors=3, rgb_range=255, scale=4, reduction=16,
                 n_resgroups=10)

    load_rcan_network('./SR_models/rcan_checkpoints/RCAN_branch1.pth', model)

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
    col = y // block_size
    row = x // block_size

    blocks_per_row = img_width // block_size

    # 计算下标
    index = row * blocks_per_row + col
    return int(index)


def main():

    patchsize = 16
    scale = 4
    seg_imgsize = 640


    lr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\LRbicx4'
    hr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\GT'

    lr_name = os.listdir(lr_path)
    lr_name.sort(key=lambda x: int(x[:-4]))


    model_seg = YOLO('yolov8n-seg.pt')

    # model
    # g_model = build_model_rcan()
    # model_sr = build_model_esrgan()
    model_sr = build_model_OmniSR()

    model_deblock = build_model_deblock()

    # psnr = 0.0
    psnr = 0.0
    ssim = 0.0
    num = 0

    # 预热
    with torch.no_grad():
        input1 = torch.randn(1, 3, 256, 256).to('cuda:0')

        # 预热
        for i in range(10):
            _ = model_sr(input1)

    TR = TimeRecorder(benchmark=False)

    for name in tqdm.tqdm(lr_name):
        lr = cv2.imread(os.path.join(lr_path, name))
        hr = cv2.imread(os.path.join(hr_path, name))
        # 1
        with torch.no_grad():
            TR.start()
            results = model_seg(lr, verbose=False, boxes=False, save=False, conf=0.39)  # list of Results objects
            TR.end_t()

        if results[0].masks == None:
            print(name)
            continue

        lr = lr.astype(np.float32)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        lr_list, num_h, num_w, h, w = crop_cpu(lr, patchsize, patchsize)
        gt_list = crop_cpu(hr, patchsize * 4, patchsize * 4)[0]
        hr = hr[:h*4, :w*4, :]

        sr_list = np.zeros((480, patchsize * scale, patchsize * scale, 3)).astype(np.float32)


        mask = results[0].masks[0].data[0].cpu().numpy()

        # s_h = mask.shape[0] / float(lr.shape[0])
        # s_w = mask.shape[1] / float(lr.shape[1])

        mask = cv2.resize(mask, (480, 270), interpolation=cv2.INTER_CUBIC)

        ### 此处比较耗时，但还可以优化
        xy = []
        for i in range(lr.shape[0]):
            for j in range(lr.shape[1]):
                if mask[int(i), int(j)] == 1:
                    xy.append([i, j])
        xy = np.stack(xy, axis=0)

        # TR.start_cpu()
        # 选择含分割点的块
        seg_block_index = set()
        for i in range(xy.shape[0]):
            index = xy2index(xy[i, 0], xy[i, 1], lr.shape[1])
            if index >= num_h*num_w:
                continue
            seg_block_index.add(index)
        seg_block_index = list(seg_block_index)

        if not seg_block_index:
            continue

        seg_block_list = []
        for i, index in enumerate(seg_block_index):
            # cv2.imwrite('./sub_imgs/%03d.png' % i, gt_list[int(index)])
            seg_block_list.append(lr_list[index])

        seg_block_array = np.stack(seg_block_list, axis=0)

        seg_block_array = seg_block_array.astype(np.float32) / 255.

        # TR.end_cpu_t()

        # 2
        with torch.no_grad():
            seg_block_tensor = torch.from_numpy(np.ascontiguousarray(seg_block_array.transpose(0, 3, 1, 2))).to('cuda:0')

            TR.start()
            sr_seg_block_tensor = model_sr(seg_block_tensor)
            TR.end_t()
            sr_seg_block = sr_seg_block_tensor.permute(0, 2, 3, 1).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

        # for i in range(sr_seg_block.shape[0]):
        #     x = sr_seg_block[i]
        #     x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('./sr_imgs_3-16/seg_esrgan_deblock/%d.png' % i, x)

        # sr_seg_block_list = [sr_seg_block[i] for i in range(sr_seg_block.shape[0])]
        # sr_seg_block_list = [cv2.cvtColor(sr_seg_block[i], cv2.COLOR_RGB2BGR) for i in range(sr_seg_block.shape[0])]
        sr_seg_block_list = [sr_seg_block[i] for i in range(sr_seg_block.shape[0])]


        # 3
        TR.start_cpu()
        # img_bic_sr = imresize_np(lr, 4, True)
        img_bic_sr = cv2.resize(lr, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        TR.end_cpu_t()

        img_bic_sr = np.uint8(np.clip(img_bic_sr, 0, 255))
        sr_list_t, _, _, _, _ = crop_cpu(img_bic_sr, patchsize*scale, patchsize*scale)

        for index, block in zip(seg_block_index, sr_seg_block_list):
            sr_list_t[index] = block

        # # 其余块插值
        # for i, block in enumerate(lr_list):
        #     if i not in seg_block_index:
        #         sr_list[i] = imresize_np(lr_list[i], 4, True)

        sr = combine(sr_list_t, num_h, num_w, h, w, patchsize, patchsize)

        sr = torch.from_numpy(np.ascontiguousarray(sr.transpose(2, 0, 1)).astype(np.float32) / 255.).unsqueeze(dim=0).to('cuda:0')

        # 4
        with torch.no_grad():
            TR.start()
            sr_deblock = model_deblock(sr)
            TR.end_t()

        sr = sr_deblock.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype(
            "uint8")

        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)


        sr_list = crop_cpu(sr, patchsize * 4, patchsize * 4)[0]
        sr_seg_block_list = []
        for i, index in enumerate(seg_block_index):
            sr_seg_block_list.append(sr_list[index])

        img_bic_sr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        sr_list_t, _, _, _, _ = crop_cpu(img_bic_sr, patchsize*scale, patchsize*scale)

        for index, block in zip(seg_block_index, sr_seg_block_list):
            sr_list_t[index] = block

        sr = combine(sr_list_t, num_h, num_w, h, w, patchsize, patchsize)

        for index in seg_block_index:
            sr_list_t[index] = gt_list[index]

        hr = combine(sr_list_t, num_h, num_w, h, w, patchsize, patchsize)

        psnr += cv2.PSNR(sr, hr)

        # 将图像转换为灰度，因为SSIM对颜色通道是分别计算的
        grayA = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)

        # 计算两个灰度图像的SSIM
        ssim_value, _ = compare_ssim(grayA, grayB, full=True)

        ssim += ssim_value

        num += 1
        TR.count()

    print('psnr: %g, ssim: %g, time: %g ms' % (psnr / num, ssim / num, TR.avg_time()))

    print('done')


if __name__ == "__main__":
    main()
