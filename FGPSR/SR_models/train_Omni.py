# from model import CS,Deep_resconsturction,CS_stage1
from ops.OmniSR import LR_SR,OmniSR,LR_SR_x8,LR_SR_INV
from ops.Recon_Net import deep_rec_dq,deep_rec,deep_rec_reuse,deep_rec_dense,deep_rec_dense_4in,deep_rec_dense_attention_3in
import torch
import numpy as np
import math
import copy
# from data import MyDataset
from torch.utils.checkpoint import checkpoint
from torch import nn, optim
from data2 import MyDataset3 as MyDataset,data_prefetcher,Test
import time

def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


if __name__ == '__main__':
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
    # net = OmniSR(kwards=kwards)

    # 判别器模型参数
    kernel_size_d = 3  # 所有卷积模块的核大小
    n_channels_d = 64  # 第1层卷积模块的通道数, 后续每隔1个模块通道数翻倍
    n_blocks_d = 8  # 卷积模块数量
    fc_size_d = 1024  # 全连接层连接数

    net_G=OmniSR(kwards=kwards).cuda()
    # SRnet = torch.load("./Net/Omni_noreuse_x4_1_16/Omni_noreuse_x4_1_16_900.pt")
    # for m in SRnet.parameters():
    #     m.requires_grad_(False)
    # net_G = LR_SR_INV(kward_lr=kwards_lr,kwards_sr=kwards).cuda()
    # net_G=LR_SR(kwards=kwards).cuda()
    # net_G.load_state_dict(torch.load("./Net/Omni4/Omni4_224.pt"))

    data_train = MyDataset()
    data_test = Test()
    train_data = torch.utils.data.DataLoader(dataset=data_train, batch_size=32, shuffle=True, num_workers=6,pin_memory=True)
    test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True, num_workers=0,
                                             pin_memory=True)
    # lossMSE = torch.nn.MSELoss()
    lossMSE = torch.nn.L1Loss()

    # lossBCE = torch.nn.BCEWithLogitsLoss()

    optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0005, weight_decay=1e-4, betas=[0.9, 0.999])
    #学习率调度器
    # step_schedule = optim.lr_scheduler.StepLR(step_size=100, gamma=0.5, optimizer=optimizer_G)
    epochs = 1000
    # vgg=vgg19().cuda()

    for i in range(0, epochs):
        # net_G.requires_grad_(True)
        psnr_list = []
        net_list = []
        if i == 250:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.00025, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 500:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.000125, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 750:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0000625, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 600:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0000625/4, weight_decay=1e-4,
                                            betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])

        sum_loss = 0
        losses = []
        t1 = time.time()
        t2 = time.time()
        save_num = int(len(train_data) / 12)
        for batch, (cropimg,sourceimg) in enumerate(train_data, start=0):
            cropimg=cropimg.cuda()
            sourceimg=sourceimg.cuda()
            ##--------生成器训练--------##
            # cropimg = SRnet(cropimg)
            # sourceimg=sourceimg.requires_grad_(True)
            # lrimg1,hrimg1 = checkpoint(SRnet, sourceimg, use_reentrant=True)
            #清空梯度流
            optimizer_G.zero_grad()

            #训练
                #裁剪图片
            # lrsize = 48
            # scales = 4
            # hrsize = lrsize*scales
            # rand1 = torch.randint(0, 64 - lrsize, size=[1])
            # rand2 = torch.randint(0, 64 - lrsize, size=[1])
            # cropimg = cropimg[:, :, rand1:rand1 + lrsize, rand2:rand2 + lrsize]
            # lrimg1 = lrimg1[:, :, rand1:rand1 + lrsize, rand2:rand2 + lrsize]
            # sourceimg = sourceimg[:, :, rand1 * scales:rand1 * scales + hrsize, rand2 * scales:rand2 * scales + hrsize]

            hr_img=net_G(cropimg)
            # print(hrimg.shape,cropimg.shape)
            # hr_img = net_G(hrimg,lrimg,SRnet.layer1)
            # hr_img = net_G(hrimg, denoise(lrimg), SRnet.layer1, SRnet.layer2)
            # hr_img = checkpoint(net_G, hrimg, cropimg, SRnet.layer1, SRnet.layer2, use_reentrant=True)
            # cropimg=cropimg.requires_grad_(True)
            # lr,lr_feature,hr_temp,hr = net_G(sourceimg, rev=False, cal_jacobian=False)

            # rand_noise = torch.randn(lr.shape[0], 3 * 15, lr.shape[2], lr.shape[3]).cuda()
            # # print(lr.shape,rand_noise.shape)
            # rand_noise = torch.cat((lr, rand_noise), 1)
            # print(lr.shape,hr.shape)

            # net_G.eval()
            # with torch.no_grad():
            #     hr_temp=net_G.layer1(rand_noise,rev=True)
            # hr_temp=hr_temp.requires_grad_(True)
            # net_G.train()


            loss1 = lossMSE(hr_img, sourceimg)
            # loss2 = lossMSE(SRnet.layer1(hr_img),lrimg)
            # loss1 = lossMSE(lr, cropimg)
            # loss2 = lossMSE(hr, sourceimg)
            # loss3 = torch.sum(lr_feature ** 2) / lr_feature.shape[0]
            # loss4 = lossMSE(hr_temp,sourceimg)

            # loss_G=loss1/64+loss3/64+loss4/16+loss2
            loss_G=loss1
            sum_loss += loss_G

            loss_G.backward()
            # print("loss_G:", loss1.item(), loss3.item(), loss4.item())
            optimizer_G.step()

            #判断最高psnr 并保存
            if batch%save_num==0 or cropimg is None:
                psnr1_sum=0
                # net_list.append(copy.deepcopy(net_G).cpu())
                # a=LR_SR_INV(kward_lr=kwards_lr, kwards_sr=kwards)
                # a.load_state_dict(net_G.state_dict())
                # a.eval()
                a=copy.deepcopy(net_G.state_dict())
                net_list.append(a)
                # net_G.cuda()
                for test_batch, (lrimg,hrimg) in enumerate(test_data, start=0):
                    lrimg=lrimg.cuda()
                    hrimg=hrimg.cuda()
                    # lrimg2,hrimg2 = checkpoint(SRnet,hrimg, use_reentrant=True)
                    # hrimg5 = checkpoint(SRnet4, denoise(lrimg2), use_reentrant=True)
                    hr_img = checkpoint(net_G, lrimg, use_reentrant=True)
                    # _,hr_img = net_G(hrimg)
                    i1 = hrimg.cpu().detach().numpy()[0]
                    i2 = hr_img.cpu().detach().numpy()[0]
                    # i1 = (i1 + 1.0) / 2.0
                    i1 = np.clip(i1, 0.0, 1.0)
                    # i2 = (i2 + 1.0) / 2.0
                    i2 = np.clip(i2, 0.0, 1.0)

                    i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
                    i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16
                    psnr1 = psnr_get(i2, i1)
                    psnr1_sum += psnr1
                psnr1_sum=psnr1_sum/(test_batch+1)
                psnr_list.append(psnr1_sum)

        psnr_max=max(psnr_list)
        index=psnr_list.index(psnr_max)
        if (i + 1) % 1 == 0:
            torch.save(net_list[index], './Net/Omni12/Omni12_{0}.pt'.format(i + 1))
            del net_list
        print('{2}|{3}    avg_loss={0}    time={1}min   psnr:{4}'.format(sum_loss / batch, (time.time() - t1) / 60, i + 1,epochs, psnr_max),index,psnr_list)
        str_list = [str(item) for item in psnr_list]
        # 使用join方法将新列表中的元素连接成一个字符串，元素之间由空格分隔
        str_list = ' '.join(str_list)
        str_write='{0}|{1}    avg_loss={2}    time={3}min   psnr_max:{4}   index={5}'.format(i + 1,epochs, sum_loss / batch, (time.time() - t1) / 60,  psnr_max,index)+'  '+str_list+'\n'
        fp = open('./Net/Omni12/Omni12.txt', 'a+')
        fp.write(str_write)
        fp.close()