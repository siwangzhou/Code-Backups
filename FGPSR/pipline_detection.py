import torch
import numpy as np
import cv2
from my_deblocking_fliter import deblocking_h265, deblocking_weight_fusion
from time_cal import TimeRecorder
import os
from ultralytics import YOLO
import math
import model
from utils import load_pretrained_state_dict
import torch.nn as nn
import argparse
import yaml
from torchvision import utils as vutils
from imgproc import tensor_to_image

from my_deblocking_fliter import deblocking_h265, deblocking_weight_fusion
import tqdm
import time
from deblocking_net.debloc_net import deblock_net

from time_cal import TimeRecorder

from MobileNetV3_deblock import MobileNetV3_deblock
from DeblockGhostnet import DeblockGhostNet
from data_util import imresize_np

from unet_deblock import deblock_unet

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

def main():

    lr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\LRbicx4'
    hr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\GT'

    imgs_name = os.listdir(lr_path)
    imgs_name.sort(key=lambda x: int(x[:-4]))

    model_det = YOLO('yolov8n.pt')
    # model_sr = build_model_esrgan()
    model_sr = build_model_OmniSR()
    model_deblock = build_model_deblock()


    psnr = 0.0
    num = 0

    # 预热
    with torch.no_grad():
        input1 = torch.randn(1, 3, 16, 16).to('cuda:0')

        # 预热
        for i in range(10):
            model_sr(input1)


    TR = TimeRecorder(benchmark=False)

    # for name in tqdm.tqdm(imgs_name):
    for name in imgs_name:
        lr = cv2.imread(os.path.join(lr_path, name))
        hr = cv2.imread(os.path.join(hr_path, name))

        # 1
        with torch.no_grad():
            TR.start()
            results = model_det(lr, verbose=False, save=False, conf=0.39)  # list of Results objects
            TR.end_t()

        boxes = results[0].boxes.data.cpu().numpy()

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(np.ascontiguousarray((lr.astype(np.float32) / 255.).transpose(2, 0, 1))).unsqueeze(dim=0).to('cuda:0')

        if boxes.shape[0] == 0:
            print(name)

        if boxes.shape[0] != 0:

            boxes = [boxes[i] for i in range(boxes.shape[0])]

            boxes.sort(key=lambda x: x[4])

            # # for box_id in range(boxes.shape[0]):
            left, top, right, bottom, _, _ = boxes[-1]
            #
            #
            left, top, right, bottom = math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
            # # box_ori = (left, top, right, bottom)
            # # left, top, right, bottom = left-2, top-2, right+2, bottom+2
            #
            #
            # left, top, right, bottom = math.floor(left)-1, math.floor(top)-1, math.ceil(right)+1, math.ceil(bottom)+1
            #
            # if left < 0 or top < 0 or right >= img.shape[1] or bottom >= img.shape[0]:
            #     continue


            obj = lr_tensor[:, :, top:bottom, left:right]
            # obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            # obj_tensor = torch.from_numpy(obj).cuda().unsqueeze(dim=0).permute(0, 3, 1, 2)

            with torch.no_grad():
                # 2
                TR.start()
                obj_sr = model_sr(obj)
                TR.end_t()

                # obj_sr = tensor_to_image(obj_sr_tensor, False, False)
                # obj_sr = cv2.cvtColor(obj_sr, cv2.COLOR_RGB2BGR)




            # 插值
            # img_bic_sr = nn.Upsample(scale_factor=4, mode='bicubic')


            # 3
            TR.start_cpu()
            # img_bic_sr = imresize_np(lr, 4, True)
            img_bic_sr = cv2.resize(lr, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            TR.end_cpu_t()
            img_bic_sr = torch.from_numpy(np.ascontiguousarray((img_bic_sr.astype(np.float32) / 255.).transpose(2, 0, 1))).unsqueeze(dim=0).to(
                'cuda:0')

            # cv2.imshow('12', img_bic_sr)
            #
            # psnr += cv2.PSNR(img_bic_sr, hr)



            # 融合
            box = (left*4, top*4, right*4, bottom*4)
            img_bic_sr[:, :, box[1]:box[3], box[0]:box[2]] = obj_sr
            # cv2.imwrite('./data/no_deblocking/'+name, img_bic_sr)



            ## 去块效应

            # start = time.time()
            # # img_sr_deblock = deblocking_h265(img_bic_sr, box)
            # # img_sr_deblock = deblocking_weight_fusion(img_bic_sr, obj_sr, box_ori, 8)
            # end = time.time()
            #
            # all_time += end - start

            # img_bic_sr = torch.from_numpy(np.ascontiguousarray(img_bic_sr/255.)).permute(2, 0, 1).float().unsqueeze(dim=0).cuda()

            # 4
            with torch.no_grad():
                TR.start()
                img_sr_deblock = model_deblock(img_bic_sr)
                # img_sr_deblock = img_bic_sr
                TR.end_t()

            img_sr_deblock = img_sr_deblock.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype("uint8")
            img_sr_deblock = cv2.cvtColor(img_sr_deblock, cv2.COLOR_RGB2BGR)
            # t = img_sr_deblock[box[1]:box[3], box[0]:box[2], :]
            # img_sr_deblock[box[1]:box[3], box[0]:box[2], :] = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)

            # cv2.imwrite('./sr_imgs_3-16/OmiSR/det_OmiSR_deblock/%s' % name, img_sr_deblock)
            # cv2.imwrite('./sr_imgs_4-28/Omni-SR/det_Omni-SR_deblock/%s' % name, img_sr_deblock)
            # cv2.imwrite('./data/deblocking_fusion/'+name, img_sr_deblock)

            psnr += cv2.PSNR(img_sr_deblock, hr)

            num += 1
            TR.count()

    print('psnr: %g, time: %g ms' % (psnr / num, TR.avg_time()))
    # print('done')


if __name__ == "__main__":
    main()