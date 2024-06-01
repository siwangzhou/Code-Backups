import os
import argparse
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log10
from PIL import Image
import numpy as np
import random

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.dataset import UCF101Dataset
from model import basicVSR
from loss import CharbonnierLoss, SSIM, FocalLoss, CrossEntropyLoss
from utils import resize_sequences
from torch.cuda.amp import autocast, GradScaler

from models.r3d_model import R3DClassifier
from models.c3d_network import C3D
from models.R2Plus1D_model import R2Plus1DClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', default='D:/DATASETS/REDS_BasicVSR/hr-set')
parser.add_argument('--lq_dir', default='D:/DATASETS/REDS_BasicVSR/lr-set')
parser.add_argument('--log_dir', default='./log_dir/20240420/')
parser.add_argument('--spynet_pretrained', default='./spynet_20210409-c6c1bd09.pth')
parser.add_argument('--scale_factor', default=4,type=int)
parser.add_argument('--batch_size', default=4,type=int)
parser.add_argument('--patch_size', default=64,type=int)
parser.add_argument('--epochs', default=500,type=int)
parser.add_argument('--num_input_frames', default=16,type=int)
parser.add_argument('--val_interval', default=1,type=int)
parser.add_argument('--max_keys', default=270,type=int)
parser.add_argument('--filename_tmpl', default='{:08d}.jpg')
parser.add_argument('--checkpoint', default='./log_dir/20240329/models/model_342.pth', type=str)
# parser.add_argument('--checkpoint', default='./log_dir/models/model_300.pth', type=str)

def train_and_test(args):
    # train_set = REDSDataset(args.gt_dir, args.lq_dir, args.scale_factor, args.patch_size, args.num_input_frames,
    #                         is_test=False, max_keys=args.max_keys, filename_tmpl=args.filename_tmpl)
    # val_set = REDSDataset(args.gt_dir, args.lq_dir, args.scale_factor, args.patch_size, args.num_input_frames,
    #                       is_test=True, max_keys=args.max_keys, filename_tmpl=args.filename_tmpl)

    gt_dir = 'D:/DataSets/UCF101/images/train/gt_256/'
    lq_dir = 'D:/DataSets/UCF101/images/train/gt_64/'
    label_path = "E:/LiuJia/Data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
    train_set = UCF101Dataset(gt_dir, lq_dir, label_path, patch_size=64)
    # train_loader = DataLoader(ucf101_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    test_gt_dir = 'D:/DataSets/UCF101/images/test/gt_256/'
    test_lq_dir = 'D:/DataSets/UCF101/images/test/gt_64/'
    val_set = UCF101Dataset(test_gt_dir, test_lq_dir, label_path, patch_size=64)
    # test_loader = DataLoader(test_ucf101_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, pin_memory=True)

    model = basicVSR(spynet_pretrained=args.spynet_pretrained).cuda()

    class_model = R3DClassifier(num_classes=101, layer_sizes=(2, 2, 2, 2))
    # class_model = C3D(num_classes=101)
    # class_model =  R2Plus1DClassifier(101, (2, 2, 2, 2), pretrained=False)
    class_model = class_model.to('cuda:0')
    # class_model = torch.load('E:/LiuJia/VPSR/log/20240418/R2Plus1D_best.pth')
    class_model = torch.load('E:/LiuJia/VPSR/log/20240417/R3D_best.pth')
    class_model.eval()

    criterion = CharbonnierLoss().cuda()
    criterion_ce = nn.CrossEntropyLoss(reduction='none').cuda()
    # criterion_ce = CrossEntropyLoss().cuda()
    criterion_mse = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam([
        {'params': model.spynet.parameters(), 'lr': 2.5e-5},
        {'params': model.backward_resblocks.parameters()},
        {'params': model.forward_resblocks.parameters()},
        {'params': model.fusion.parameters()},
        {'params': model.upsample1.parameters()},
        {'params': model.upsample2.parameters()},
        {'params': model.conv_hr.parameters()},
        {'params': model.conv_last.parameters()}
    ], lr=2e-5, betas=(0.9, 0.99)
    # ], lr=2e-6, betas=(0.9, 0.999)
    )

    max_epoch = args.epochs
    total_steps = max_epoch * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-7)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #                 optimizer, lambda step: 1 - step / total_steps)
    scaler = GradScaler()

    os.makedirs(f'{args.log_dir}/models', exist_ok=True)
    os.makedirs(f'{args.log_dir}/images', exist_ok=True)
    train_loss = []
    validation_loss = []
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']+1
        print(f"load checkpoint from {start_epoch-1}")

    for epoch in range(start_epoch, max_epoch):
        model.train()
        # fix SPyNet and EDVR at first 5000 iteration
        if epoch < 150:
            for k, v in model.named_parameters():
                if 'spynet' in k or 'edvr' in k:
                    v.requires_grad_(False)
        elif epoch == 150:
            # train all the parameters
            model.requires_grad_(True)

        epoch_loss = 0
    #     with tqdm(train_loader, ncols=100) as pbar:
    #         for idx, data in enumerate(pbar):
    #             gt_sequences, lq_sequences, labels = Variable(data[0]), Variable(data[1]), Variable(data[2])
    #             gt_sequences = gt_sequences.to('cuda:0')
    #             lq_sequences = lq_sequences.to('cuda:0')
    #             labels = torch.zeros_like(labels)  # 对抗到第0类
    #             # labels = torch.randint_like(labels, 101)
    #             labels = torch.LongTensor(labels)
    #             labels = labels.to('cuda:0')  # 无目标对抗
    #
    #             with autocast():
    #                 pred_sequences = model(lq_sequences)
    #                 loss_sr = criterion(pred_sequences, gt_sequences)
    #
    #                 pred_sequences_cls = pred_sequences.clone()
    #                 pred_sequences_cls = pred_sequences_cls.transpose(1, 2)
    #                 pred_cls = class_model(pred_sequences_cls)
    #                 # print(pred_cls)
    #                 # pred_cls = torch.nn.functional.sigmoid(pred_cls)
    #                 # pred_cls = torch.clamp(pred_cls, min=1e-7, max=1 - 1e-7)
    #                 loss_cls = criterion_ce(pred_cls, labels)
    #                 # nan_mask = torch.isnan(loss_cls)
    #                 # if sum(nan_mask) > 0:
    #                 #     if sum(~nan_mask) == 0:
    #                 #         loss_cls = torch.tensor(0.0, requires_grad=True)
    #                 #     else:
    #                 #         loss_cls = loss_cls.masked_select(~nan_mask).mean()
    #                 # else:
    #                 loss_cls = loss_cls.mean()
    #
    #
    #                 # loss_cls_log = torch.log(1 + abs(10.0 - loss_cls))
    #
    #                 # if not torch.isnan(loss_cls_log):
    #                 loss = loss_sr + 0.01*loss_cls
    #                 # else:
    #                 #     loss = loss_sr
    #
    #                 epoch_loss += loss.item()
    #
    #             optimizer.zero_grad()
    #             scaler.scale(loss).backward()
    #             # scaler.scale(loss_cls_log).backward()
    #             # loss.backward()
    #             # optimizer.step()
    #             # nn.utils.clip_grad_norm_(class_model.parameters(), max_norm=20, norm_type=2)
    #             scaler.step(optimizer)
    #             scaler.update()
    #             scheduler.step()
    #
    #             pbar.set_description(f'[Epoch {epoch}]')
    #             pbar.set_postfix(OrderedDict(loss=f'{loss.data:.3f}',
    #                                          loss_sr=f'{loss_sr.data:.3f}',
    #                                          loss_cls=f'{loss_cls.data:.3f}'),
    #                                          lr=optimizer.state_dict()['param_groups'][0]['lr'])
    #
    #         train_loss.append(epoch_loss / len(train_loader))
    #
    #     if (epoch) % args.val_interval != 0:
    #         continue
    #
    #     model.eval()
    #     val_loss = 0
    #     val_psnr, lq_psnr = 0, 0
    #     val_acc, lq_acc = 0, 0
    #     os.makedirs(f'{args.log_dir}/images/epoch{epoch:05}', exist_ok=True)
    #     with torch.no_grad():
    #         for idx, data in enumerate(val_loader):
    #             gt_sequences, lq_sequences, labels = data
    #             gt_sequences = gt_sequences.to('cuda:0')
    #             lq_sequences = lq_sequences.to('cuda:0')
    #             # labels = torch.zeros_like(labels) # 对抗到第0类
    #             labels = torch.LongTensor(labels)
    #             labels = labels.to('cuda:0')  # 无目标对抗
    #             pred_sequences = model(lq_sequences)
    #
    #             loss_2 = criterion(pred_sequences,gt_sequences)
    #             val_loss += loss_2.item()
    #
    #             lq_sequences = resize_sequences(lq_sequences, (gt_sequences.size(dim=3), gt_sequences.size(dim=4)))
    #             val_mse = criterion_mse(pred_sequences, gt_sequences)
    #             lq_mse = criterion_mse(lq_sequences, gt_sequences)
    #             val_psnr += 10 * log10(1 / val_mse.data)
    #             lq_psnr += 10 * log10(1 / lq_mse.data)
    #
    #             pred_sequences_to_cls = pred_sequences.transpose(1, 2)
    #             outputs = class_model(pred_sequences_to_cls)
    #             preds = torch.max(outputs, 1)[1]
    #             val_acc += torch.sum(preds == labels.data)
    #             lq_sequences_to_cls = lq_sequences.transpose(1, 2)
    #             outputs = class_model(lq_sequences_to_cls)
    #             preds = torch.max(outputs, 1)[1]
    #             lq_acc += torch.sum(preds == labels.data)
    #
    #
    #             save_image(pred_sequences[0], f'{args.log_dir}/images/epoch{epoch:05}/{idx}_SR.jpg', nrow=4)
    #             save_image(lq_sequences[0], f'{args.log_dir}/images/epoch{epoch:05}/{idx}_LQ.jpg', nrow=4)
    #             save_image(gt_sequences[0], f'{args.log_dir}/images/epoch{epoch:05}/{idx}_GT.jpg', nrow=4)
    #
    #         validation_loss.append(val_loss / len(val_loader))
    #
    #     print(f'==[validation]== PSNR:{val_psnr / len(val_loader):.2f},(lq:{lq_psnr / len(val_loader):.2f})')
    #     print(f'==[validation]== ACC:{val_acc / len(val_loader):.2f},(lq:{lq_acc / len(val_loader):.2f})')
    #     with open(f'{args.log_dir}/validation_log.txt', 'a+') as log_file:
    #         log_file.write(f'==[validation{epoch}]== PSNR:{val_psnr / len(val_loader):.2f},(lq:{lq_psnr / len(val_loader):.2f})' + '\n')
    #         log_file.write(
    #             f'==[validation{epoch}]== ACC:{val_acc / len(val_loader):.2f},(lq:{lq_acc / len(val_loader):.2f})' + '\n')
    #
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'scheduler_state_dict': scheduler.state_dict(),
    #         'scaler_state_dict': scaler.state_dict()
    #     }, f'{args.log_dir}/models/model_{epoch}.pth')
    #
    # fig = plt.figure()
    # train_loss = [loss for loss in train_loss]
    # validation_loss = [loss for loss in validation_loss]
    # x_train = list(range(len(train_loss)))
    # x_val = [x for x in range(max_epoch) if (x + 1) % args.val_interval == 0]
    # plt.plot(x_train, train_loss)
    # plt.plot(x_val, validation_loss)
    #
    # fig.savefig(f'{args.log_dir}/loss.png')

    # 测试
    model.eval()
    val_loss = 0
    val_psnr, lq_psnr = 0, 0
    val_ssim, lq_ssim = 0, 0
    val_acc, lq_acc = 0, 0


    os.makedirs(f'{args.log_dir}/Non_AP_SR', exist_ok=True)
    # os.makedirs(f'{args.log_dir}/LR', exist_ok=True)
    # os.makedirs(f'{args.log_dir}/HR', exist_ok=True)
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            print("Processing Image-", idx)
            gt_sequences, lq_sequences, labels = data
            gt_sequences = gt_sequences.to('cuda:0')
            lq_sequences = lq_sequences.to('cuda:0')
            # labels = torch.zeros_like(labels)
            labels = labels.to('cuda:0')
            # with autocast():
            pred_sequences = model(lq_sequences)

            myssim = SSIM()

            lq_sequences = resize_sequences(lq_sequences, (gt_sequences.size(dim=3), gt_sequences.size(dim=4)))
            # val_mse = criterion_mse(pred_sequences, gt_sequences)
            # lq_mse = criterion_mse(lq_sequences, gt_sequences)
            # val_psnr += 10 * log10(1 / val_mse.data)
            # lq_psnr += 10 * log10(1 / lq_mse.data)
            # lq_ssim += myssim(gt_sequences, lq_sequences)
            # val_ssim += myssim(gt_sequences, pred_sequences)

            for i in range(len(pred_sequences[0])):
                save_image(pred_sequences[0][i], f'{args.log_dir}/Non_AP_SR/{idx}_{i}.jpg')
                # save_image(lq_sequences[0][i], f'{args.log_dir}/LR/{idx}_{i}.jpg')
                # save_image(gt_sequences[0][i], f'{args.log_dir}/HR/{idx}_{i}.jpg')

            pred_sequences_to_cls = pred_sequences.transpose(1, 2)
            lq_sequences_to_cls = lq_sequences.transpose(1, 2)
            with autocast():
                outputs = class_model(pred_sequences_to_cls)
                preds = torch.max(outputs, 1)[1]
            val_acc += torch.sum(preds == labels.data)
            with autocast():
                outputs = class_model(lq_sequences_to_cls)
                preds = torch.max(outputs, 1)[1]
            lq_acc += torch.sum(preds == labels.data)

    # print(f'==[validation]== PSNR:{val_psnr / len(val_loader):.4f},(lq:{lq_psnr / len(val_loader):.4f})')
    # print(f'==[validation]== SSIM:{val_ssim / len(val_loader):.4f},(lq:{lq_ssim / len(val_loader):.4f})')
    print(f'==[validation]== Acc:{val_acc / len(val_loader):.4f},(lq:{lq_acc / len(val_loader):.4f})')




torch.manual_seed(10)
random.seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)
if __name__ == '__main__':
    args = parser.parse_args()
    train_and_test(args)