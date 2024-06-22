# -*- coding: utf-8 -*-
"""
# @file name  : .py
# @author     : 
# @date       : 
# @brief      : 
"""
import argparse
import os
import torch.utils.data
import torchvision.utils as vutils
from data_loader import get_loader
from model import generate, discriminator, Generator

parser = argparse.ArgumentParser()
# D:\Baoxiu\datas\image E:\BaoXiu\data\crop178_resize224_1368
parser.add_argument('--data_address', default='E:\BaoXiu\data\datas\dd\img_align_celeba')
parser.add_argument('--attr_address', default=r'E:\BaoXiu\data\datas\Anno\list_attr_celeba.txt')
parser.add_argument('--netG', default=r"E:\BaoXiu\EX\newDATNet\out-G_IN-D_IN-patchGAN-b_0.3\model\G/netG_iters_072000.pth", help="path to netG (to continue training)")
parser.add_argument('--atts', default=('Male', 'Young'), help='male or female or all')
parser.add_argument('--out_dir', default='./G_IN-D_IN-b_0.3-iters-072000')

opt = parser.parse_args()
device = 'cuda'

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

# 读入数据集
test_loader = get_loader(opt.data_address, opt.attr_address, 1, 'CelebA', 'evaluation', 0, opt.atts)

# # 临时的加载  可随时删除
# netG = Generator(64)
# netG.to(device)
# state_dic = torch.load(opt.netG, map_location=torch.device('cuda'))
# netG.load_state_dict(state_dic['state_dict'])
# netG.eval()


# 初始化模型
netG = generate(path=opt.netG, layers=64, norm_layer='IN')
netG.to(device)
netG.eval()

print("starting...")
i = 200799

for x, y, name in test_loader:
    i += 1
    x = x.cuda()
    pertur = netG(x)
    fake = torch.tanh(pertur + x)
    vutils.save_image(fake, opt.out_dir + '/%06d.png' % (name.item()), value_range=(-1, 1), normalize=True)
