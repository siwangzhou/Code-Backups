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
    col = x // block_size
    row = y // block_size

    blocks_per_row = img_width // block_size

    # 计算下标
    index = row * blocks_per_row + col
    return index


def main():

    patchsize = 16

    # lr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\LRbicx4'
    lr_path = r'G:\BaoXiu\EX\ESRGAN-PyTorch-main\sr_imgs_3-16\OmiSR\block_OmiSR'
    hr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\GT'

    lr_name = os.listdir(lr_path)
    lr_name.sort(key=lambda x: int(x[:-4]))
    # seg_model = YOLO('yolov8n-seg.pt')

    # model
    # g_model = build_model_rcan()
    g_model = build_model_esrgan()

    model_deblock = build_model_deblock()

    # psnr = 0.0
    psnr = 0.0
    num = 0

    TR = TimeRecorder(benchmark=False)

    for name in tqdm.tqdm(lr_name):
        num += 1

        lr = cv2.imread(os.path.join(lr_path, name))
        hr = cv2.imread(os.path.join(hr_path, name))

        # lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        # lr_list, num_h, num_w, h, w = crop_cpu(lr, patchsize, patchsize)
        # gt_list = crop_cpu(hr, patchsize * 4, patchsize * 4)[0]
        # hr = hr[:h*4, :w*4, :]
        hr = hr[:1024, :1920, :]


        # lr_array = np.stack(lr_list, axis=0)

        # results = seg_model(lr, verbose=False, boxes=False, save=False, conf=0.39)  # list of Results objects
        # xy = results[0].masks.xy[0]
        #
        # seg_block_index = set()
        # for i in range(xy.shape[0]):
        #     index = xy2index(xy[i, 0], xy[i, 1], lr.shape[1])
        #     if index >= num_h*num_w:
        #         continue
        #     seg_block_index.add(index)
        #
        # for i, index in enumerate(seg_block_index):
        #     cv2.imwrite('./sub_imgs/%03d.png' % i, gt_list[int(index)])


        lr_array = lr.astype(np.float32) / 255.
        # lr_array = lr_array.astype(np.float32) / 255.


        # lr_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))).float().unsqueeze(dim=0).cuda()

        # with torch.no_grad():
        #     lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_array.transpose(0, 3, 1, 2))).to('cuda:0')
        #     # lr_tensor=torch.randn(480, 3, 16, 16).to('cuda:0')
        #
        #     TR.start()
        #     sr_tensor = g_model(lr_tensor)
        #     TR.end()
            # sr = tensor2img(sr_tensor.detach()[0].float().cpu(), out_type=np.uint8, min_max=(0, 255))  # uint8

            # sr = sr_tensor.permute(0, 2, 3, 1).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

        # sr_list = [sr[i] for i in range(sr.shape[0])]
        #
        # sr = combine(sr_list, num_h, num_w, h, w, patchsize, patchsize)
        # sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        #
        # psnr += cv2.PSNR(sr, hr)
        #
        # cv2.imwrite('./sr_imgs_3-16/block_esrgan_parallel/%s' % name, sr)

    # print('psnr: %g, time: %g ms' % (psnr / num, TR.avg_time_pre()))
    #
    # print('done')

    # print(1)
        with torch.no_grad():
            # cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
            # sr = torch.from_numpy(np.ascontiguousarray(sr / 255.)).permute(2, 0, 1).float().unsqueeze(dim=0).cuda()
            lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_array.transpose(2, 0, 1))).unsqueeze(dim=0).to('cuda:0')
            TR.start()
            sr_deblock = model_deblock(lr_tensor)
            TR.end()

            sr_deblock = sr_deblock.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
            sr_deblock = cv2.cvtColor(sr_deblock, cv2.COLOR_RGB2BGR)

            # psnr += cv2.PSNR(sr_deblock, hr)
            psnr += cv2.PSNR(lr, hr)

            # cv2.imwrite('./sr_imgs_3-16/OmiSR/block_OmiSR_deblock/%s' % name, sr_deblock)

    print('psnr: %g, time: %g ms' % (psnr / num, TR.avg_time_pre()))
        # sr = tensor2img(sr_tensor.detach()[0].float().cpu())
                # sr = tensor2img(sr_tensor.detach()[0].float().cpu(), out_type=np.uint8, min_max=(0, 255))  # uint8
                # sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

                # cv2.imwrite('./rcan_sr_imgs_64/'+name, sr)
                # print(1)

                # cv2.imshow('12', sr)
                # cv2.waitKey(0)

                # psnr += cv2.PSNR(img_bic_sr, hr)

                ## 去块效应

                # start = time.time()
                # # img_sr_deblock = deblocking_h265(img_bic_sr, box)
                # # img_sr_deblock = deblocking_weight_fusion(img_bic_sr, obj_sr, box_ori, 8)
                # end = time.time()
                #
                # all_time += end - start


        #
        #         img_sr_deblock = img_sr_deblock.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype("uint8")
        #
        #         cv2.imwrite('./data/deblock_unet_1_cen/'+name, img_sr_deblock)
        #         # cv2.imwrite('./data/deblocking_fusion/'+name, img_sr_deblock)
        #
        #         psnr += cv2.PSNR(img_sr_deblock, hr)
        #         num += 1
        #
        # psnr = psnr / num
        # # time = all_time / num
        # time = TR.avg_time()
        #
        # print(psnr)
        # print(time)
        # print(num)


if __name__ == "__main__":
    main()
