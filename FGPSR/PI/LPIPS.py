import torch
import lpips
from PIL import Image
import numpy as np
import os
import torch

from tqdm import tqdm


gt_dir = [r'G:\BaoXiu\EX\data\vedio_process\data\dance\GT', r'G:\BaoXiu\EX\data\vedio_process\data\dance\GT_crop']

# dir_paths = ['G:\BaoXiu\EX\ESRGAN-PyTorch-main\sr_imgs_4-28\ESRGAN', 'G:\BaoXiu\EX\ESRGAN-PyTorch-main\sr_imgs_4-28\Omni-SR']
dir_paths = [r'G:\BaoXiu\EX\ESRGAN-PyTorch-main\sr_imgs_4-28\temp']

# 加载预训练的LPIPS模型
lpips_model = lpips.LPIPS(net="alex", verbose=False, spatial=True).cuda()

for path in dir_paths:

    dir_list = os.listdir(path)

    for sub_path in dir_list:
        test_path = os.path.join(path, sub_path)

        test_list = os.listdir(test_path)

        Lpips=0

        for imgname in tqdm(test_list):
            # for imgname in test_list:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            img_str = test_path + '/' + imgname
            image1 = Image.open(img_str)

            if image1.size[1] == 1024:
                gt_str = gt_dir[1] + '/' + imgname
            else:
                gt_str = gt_dir[0] + '/' + imgname

            image2 = Image.open(gt_str)

            # 将图像转换为PyTorch的Tensor格式
            image1_tensor = lpips.im2tensor(lpips.load_image(gt_str)).cuda()
            image2_tensor = lpips.im2tensor(lpips.load_image(img_str)).cuda()


            # 使用LPIPS模型计算距离
            distance = lpips_model.forward(image1_tensor, image2_tensor)

            Lpips += distance.mean().item()

        print("%s , LPIPS %g:" % (sub_path, Lpips / len(test_list)))




