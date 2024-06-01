# -*- coding: utf-8 -*-
# @Time : 2021/11/24 0024 12:19
# @Author : ZeroOne
# @Email : hnuliujia@hnu.cn
# @File : test_lpips.py


import torch
import lpips
# from IPython import embed
import os
from torchvision import transforms as T
from PIL import Image


def img2Tensor(path):
    image = Image.open(path).convert('L')
    transform = []
    # transform.append(T.CenterCrop(180))
    # transform.append(T.Resize(224))
    transform.append(T.ToTensor())
    transform.append(T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    # transform.append(T.Normalize((0.5, ), (0.5, )))
    transform = T.Compose(transform)
    image = transform(image)
    image = torch.reshape(image, [1,3,256,256])
    return image

use_gpu = True  # Whether to use GPU
spatial = True  # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='vgg', spatial=spatial)  # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=True) # Can also set net = 'squeeze' or 'vgg'

if (use_gpu):
    loss_fn.cuda()

## Example usage with dummy tensors
# root_path = 'D:/MNIST/Adv_Rank_5/'
# im0_path_list = ['{}.png'.format(i) for i in range(1000)]
# im1_path_list = []
# for root, fnames, _ in sorted(os.walk(root_path, followlinks=True)):
#     i = 0
#     for fname in fnames:
#         dir = os.path.join(root, fname)
#         if i == 0:
#             for path in os.listdir(dir):
#                 im0_path_list.append(os.path.join(dir,path))
#         else:
#             for path in os.listdir(dir):
#                 im1_path_list.append(os.path.join(dir, path))
#         i = i+1
        # if '_generated' in fname:
        #     im0_path_list.append(path)
        # elif '_real' in fname:
        #     im1_path_list.append(path)

im0_path_list = ['E:/LiuJia/VPSR/log_dir/20240420/BasicVSR/HR/' + filename for filename in os.listdir('E:/LiuJia/VPSR/log_dir/20240420/BasicVSR/HR/')]
im1_path_list = ['E:/LiuJia/VPSR/log_dir/20240420/Non_AP_SR/' + filename for filename in os.listdir('E:/LiuJia/VPSR/log_dir/20240420/Non_AP_SR/')]

import copy
def getListMaxNumIndex2(num_list,topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    tmp_list=copy.deepcopy(num_list)
    tmp_list.sort()
    # max_num_index=[num_list.index(one) for one in tmp_list[::-1][:topk]]
    min_num_index=[num_list.index(one) for one in tmp_list[:topk]]
    return [num_list[i]for i in min_num_index], min_num_index

dist_ = []
for i in range(len(im0_path_list)):
    dummy_im0 = lpips.im2tensor(lpips.load_image(im0_path_list[i]))
    dummy_im1 = lpips.im2tensor(lpips.load_image(im1_path_list[i]))
    # dummy_im0 = img2Tensor(root_path + im0_path_list[i])
    # dummy_im1 = img2Tensor(r'D:\MNIST\Original\5\49.png')
    if (use_gpu):
        dummy_im0 = dummy_im0.cuda()
        dummy_im1 = dummy_im1.cuda()
    dist = loss_fn.forward(dummy_im0, dummy_im1)
    dist_.append(dist.mean().item())
print('Avarage Distances: %.4f' % (sum(dist_) / len(im0_path_list)))

values, index = getListMaxNumIndex2(dist_, topk=10)
print('Names:', [im0_path_list[j]for j in index])
print('Values:', values)