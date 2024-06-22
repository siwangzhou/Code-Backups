from __future__ import print_function
import datetime
import random
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from model import generate, discriminator, ContentLoss, FaceMatchingLoss, GANLoss, MultiScaleDiscriminator
from efficientnet_pytorch import EfficientNet
from utils.face_lfw import face_matching
from utils.cal_acc import attrs_classification
from utils.model_cr import _G_xvz, _G_vzx
from torchvision import transforms as T


class DATNet_solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, config):
        """Initialize opturations."""
        self.config = config

        self.annotations = config.annotations

        # Data loader.
        self.celeba_loader = celeba_loader

        # out directories.
        self.out_model_G = config.out_model + 'G'
        self.out_model_D = config.out_model + 'D'
        self.out_model_C = config.out_model + 'C'
        self.out_log = config.out_log
        self.out_samples = config.out_fake

        # training configs
        self.attrs = config.attrs
        self.start_iters = config.start_iters
        self.end_iters = config.end_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.iters_lr_decay = config.iters_lr_decay

        self.alpha1 = config.alpha1
        self.alpha2 = config.alpha2
        self.beta_gender = config.beta_gender
        self.beta_age = config.beta_age

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.gamma = config.gamma
        self.a = config.a
        self.b = config.b
        self.n_g = config.n_g
        self.n_c = config.n_c

        self.log_save_step = config.log_save_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # model weights paths
        self.g_weights_path = config.g_weights_path
        self.d_weights_path = config.d_weights_path
        self.c_weights_path = config.c_weights_path

        # others
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_continue = config.is_continue
        self.start_time = time.time()
        self.current_iters = 0
        self.current_epochs = 0

        # loss function
        self.CL_da = ContentLoss('features.34').to(self.device)
        self.CL_fm = FaceMatchingLoss().to(self.device)
        self.tanh_space = nn.Tanh()
        self.criterion = nn.BCEWithLogitsLoss()

        self.gan_mode = config.gan_mode
        self.gan_loss = GANLoss(self.gan_mode).to(self.device)

        # evalution
        self.attrs_classification = attrs_classification
        self.face_matching = face_matching

        # Build the model and tensorboard.
        self.g_normalization = config.g_normalization
        self.d_normalization = config.d_normalization
        self.build_model()
        self.build_tensorboard()

        # train with cr
        self.train_with_cr = config.train_with_cr
        if self.train_with_cr:
            self.view_list = [1, 2, 3, 5, 6, 7]
            self.build_CR()

    def build_model(self):
        """Create all sub-models in DATNet."""

        self.G = generate(path=self.g_weights_path, layers=64, norm_layer=self.g_normalization)
        self.D = discriminator(path=self.d_weights_path, layers=64, norm_layer=self.d_normalization)
        self.C = EfficientNet.from_name('efficientnet-b0', num_classes=2)

        self.C.load_state_dict(torch.load(self.c_weights_path, map_location=self.device)['state_dict'])

        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)

        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.c_optimizer = optim.Adam(self.C.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        # self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr)
        # self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lr)
        # self.c_optimizer = optim.RMSprop(self.C.parameters(), lr=self.lr)



        if self.is_continue != 0:
            # 加载优化器参数
            state_dic = torch.load(self.g_weights_path, map_location=torch.device(self.device))
            self.g_optimizer.load_state_dict(state_dic['optimizer'])

            state_dic = torch.load(self.d_weights_path, map_location=torch.device(self.device))
            self.d_optimizer.load_state_dict(state_dic['optimizer'])

            state_dic = torch.load(self.c_weights_path, map_location=torch.device(self.device))
            self.c_optimizer.load_state_dict(state_dic['optimizer'])

        # # setup lr_scheduler
        # last_epoch = self.start_iters - self.iters_lr_decay - 1 if self.start_iters >= self.iters_lr_decay else -1
        # self.g_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=self.lr_update_step, gamma=0.8, last_epoch=last_epoch)
        # self.d_scheduler = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=self.lr_update_step, gamma=0.8, last_epoch=last_epoch)
        # self.c_scheduler = optim.lr_scheduler.StepLR(self.c_optimizer, step_size=self.lr_update_step, gamma=0.8, last_epoch=last_epoch)


        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')

    def build_CR(self):

        self.CRxvz = torch.nn.DataParallel(_G_xvz())
        self.CRvzx = torch.nn.DataParallel(_G_vzx())

        self.CRxvz.load_state_dict(torch.load('./weights/CR/netG_xvz.pth', map_location=self.device))
        self.CRvzx.load_state_dict(torch.load('./weights/CR/netG_vzx.pth', map_location=self.device))

        self.CRxvz.to(self.device)
        self.CRvzx.to(self.device)

        self.CRxvz.eval()
        self.CRvzx.eval()



    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from tensorboardX import SummaryWriter
        self.SW = SummaryWriter(log_dir=self.out_log)

    def get_current_lr(self):
        return self.g_optimizer.param_groups[0]['lr']

    def update_lr(self):
        """Decay learning rates of the generator and discriminator."""

        lr = self.lr * (1 - (self.current_iters + 1 - self.iters_lr_decay) / (self.end_iters - self.iters_lr_decay))

        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = lr

    def logger(self, state=1, mylog=''):
        if state == 1:
            some_args = '  lr = ' + str(self.lr) + '\n' \
                        + '  alpha_per = ' + str(self.alpha1) + '\n' \
                        + '  alpha_fm = ' + str(self.alpha2) + '\n' \
                        + '  beta_gender = ' + str(self.beta_gender) + '\n' \
                        + '  beta_age = ' + str(self.beta_age) + '\n' \
                        + '  gamma = ' + str(self.gamma) + '\n' \
                        + '  n = ' + str(self.n_c) + '\n' \
                        + '  a = ' + str(self.a) + '\n' \
                        + '  b = ' + str(self.b) + '\n'
            mylog = 'strating time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
                                                      time.localtime(time.time())) + '\n' + some_args + self.annotations + '\n'

            if self.start_iters != 0:
                mylog += '================continue training from iters %d ==============\n' % (self.start_iters + 1) + str(
                    self.config) + '\n'
                print('continue training from iters %d .........' % (self.start_iters + 1))
            else:
                mylog += str(self.config) + '\n'

            print('start training........')


        # write in text file
        with open('%s/log.txt' % self.out_log, 'a', encoding='utf-8') as f:
            f.write(mylog)
            f.close()

    # def update_lr(self):
    #     """Decay learning rates of all sub-models."""
    #     self.g_scheduler.step()
    #     self.d_scheduler.step()
    #     self.c_scheduler.step()

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def evaluation(self):
        # evaluation
        with torch.no_grad():
            self.G.eval()
            gender_acc = self.attrs_classification(self.G, data_path=self.config.data_address, attr_path=self.config.attr_address)
            age_acc = self.attrs_classification(self.G, 'age', data_path=self.config.data_address, attr_path=self.config.attr_address)
            face_match_acc = face_matching(self.G, self.current_iters + 1, data_path=self.config.data_address)
            self.G.train()

        scalars = {}
        scalars['EVA/gender'] = gender_acc
        scalars['EVA/age'] = age_acc
        scalars['EVA/face_match'] = face_match_acc

        for tag, value in scalars.items():
            self.SW.add_scalar(tag, value, self.current_iters)

        with open('%s/acc_log.txt' % self.out_log, 'a', encoding='utf-8') as f:
            f.write('iters %d / epoch %d : %g  %g  %g \n' % (self.current_iters + 1, self.current_epochs + 1, gender_acc, age_acc, face_match_acc))
            f.close()

    def save_models(self):
        # save weights
        G_state = {
            'state_dict': self.G.state_dict(),
            'optimizer': self.g_optimizer.state_dict(),
        }
        D_state = {
            'state_dict': self.D.state_dict(),
            'optimizer': self.d_optimizer.state_dict(),
        }
        M_state = {
            'state_dict': self.C.state_dict(),
            'optimizer': self.c_optimizer.state_dict(),
        }
        torch.save(G_state, '%s/G/netG_iters_%06d.pth' % (self.config.out_model, self.current_iters + 1))
        torch.save(D_state, '%s/D/netD_iters_%06d.pth' % (self.config.out_model, self.current_iters + 1))
        torch.save(M_state, '%s/C/netC_iters_%06d.pth' % (self.config.out_model, self.current_iters + 1))

    def transform_for_cr(self, x):

        x = T.Resize(178)(x)
        x = T.CenterCrop(168)(x)

        x = x[:, :, 40:, 20:148]

        return x

    def detransform_for_cr(self, x):

        tanker = torch.zeros([x.shape[0], 3, 168, 168], device=self.device)

        tanker[:, :, 40:, 20: 148] = x

        x = T.CenterCrop(178)(tanker)

        x = T.Resize(224)(x)

        return x

    def adv_multi_attrs_loss(self, y, target):
        loss = self.beta_gender * self.criterion(y[:, 0], target[:, 0]) + self.beta_age * self.criterion(y[:, 1], target[:, 1])
        return loss


    def train(self):
        """Train StarGAN within a single dataset."""
        self.logger(state=1)
        it_nums = 0

        errC = torch.zeros(1).to(self.device)

        
        data_iter = iter(self.celeba_loader)

        for self.current_iters in range(self.start_iters, self.end_iters+1):
            
            self.current_epochs = self.current_iters * self.batch_size // 200000
            
            ########### process data ##########

            # Fetch real images and labels.
            try:
                x, y, _ = next(data_iter)
            except:
                data_iter = iter(self.celeba_loader)
                x, y, _ = next(data_iter)
            
            y = y.squeeze(1)
            xr, label = x.to(self.device), y.to(self.device)

            ########### train D ##########
            self.reset_grad()
            ## train with real
            real_logits = self.D(xr)
            errD_real = self.gan_loss(real_logits, True)

            ## train with fake
            pertur = self.G(xr)
            fake = self.tanh_space(pertur + xr)
            fake_logits = self.D(fake.detach())
            errD_fake = self.gan_loss(fake_logits, False)

            if self.gan_mode == 'wgangp':
                errD_fake = torch.mean(fake_logits)
                ## compute loss for gradient penalty
                alpha = torch.rand(xr.size(0), 1, 1, 1).to('cuda')
                x_hat = (alpha * xr + (1 - alpha) * fake.detach()).requires_grad_(True)
                out_src = self.D(x_hat)
                errD_gp = self.gradient_penalty(out_src, x_hat)
                errD = errD_real + errD_fake + 10 * errD_gp
            else:
                errD = errD_real + errD_fake

            errD.backward()
            self.d_optimizer.step()

            # summarize
            scalars = {}
            scalars['D/loss'] = errD.item()
            scalars['D/loss_real'] = errD_real.item()
            scalars['D/loss_fake'] = errD_fake.item()
            if self.gan_mode == 'wgangp':
                scalars['D/loss_gp'] = errD_gp.item()

            ########### train G ##########
            if (self.current_iters + 1) % self.n_g == 0:
                # pertur = netG(xr)
                # fake = tanh(pertur + xr)

                self.reset_grad()

                # train with D
                G_fake_logits = self.D(fake)
                errG_d = self.gan_loss(G_fake_logits, True)
                # perception
                errG_percep = self.CL_da(xr, fake)
                # face matching
                errG_matching = self.CL_fm(xr, fake)

                # train with classier netC
                label_pred = self.C(fake)
                target_label = torch.zeros_like(label, device=self.device)
                target_label[:, 1] = 1
                # errG_adv = criterion(output, target_label)   # all in
                errG_adv_fake = self.adv_multi_attrs_loss(label_pred, target_label)

                # errG_adv = criterion(output, 1 - label)   # reverse
                # errG_adv = torch.zeros(1).to(device)

                if self.train_with_cr:
                    # cr
                    view_label = torch.zeros([label.shape[0],], device=self.device, dtype=torch.long)
                    view_label = torch.eye(9, device=self.device)[view_label.fill_(self.view_list[random.randint(0, 5)])]
                    v_bar, z_bar = self.CRxvz(self.transform_for_cr(fake))
                    fake_view = self.detransform_for_cr(self.CRvzx(view_label, z_bar))
                    label_pred_view = self.C(fake_view)
                    errG_adv_view = self.adv_multi_attrs_loss(label_pred_view, target_label)
                    scalars['G/loss_adv_TF'] = errG_adv_view.item()
                    # errG_adv = errG_adv_fake + self.b * errG_adv_view
                    errG_adv = self.b * errG_adv_fake + errG_adv_view
                else:
                    errG_adv = errG_adv_fake
                err = errG_percep * self.alpha1 + errG_matching * self.alpha2 + errG_adv + errG_d * self.gamma

                err.backward()
                self.g_optimizer.step()

                # summarize
                scalars['G/loss'] = err.item()
                scalars['G/loss_fea'] = errG_percep.item()
                scalars['G/loss_adv'] = errG_adv.item()
                scalars['G/loss_gan'] = errG_d.item()
                scalars['G/loss_matching'] = errG_matching.item()


            ########### train efnet-imagenet ##########
            self.reset_grad()
            if (self.current_iters + 1) % self.n_c == 0:
                # netC.train()
                M_real_logits = self.C(xr).squeeze(1)
                errC_x = self.criterion(M_real_logits, label)

                pertur = self.G(xr)
                fake = self.tanh_space(pertur + xr)
                M_fake_logits = self.C(fake.detach()).squeeze(1)
                errC_gx = self.criterion(M_fake_logits, label)
                #
                errC = self.a * errC_x + (1 - self.a) * errC_gx
                # errC = errC_gx

                errC.backward()
                self.c_optimizer.step()

                # summarize
                scalars['C/loss'] = errC.item()

                # netC.eval()

            ######################### logger ####################

            if (self.current_iters + 1) % self.log_save_step == 0:

                for tag, value in scalars.items():
                    self.SW.add_scalar(tag, value, it_nums)
                it_nums += 1

                et = time.time() - self.start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                # 控制台输出

                infos = 'Elapsed[%s] [%d/%d][%d][%f] Loss_Dis: %.4f Loss_Mat: %.4f Loss_RN: %.4f Loss_Gd: %.4f ' \
                        'Loss_D: %.4f  errC: %.4f' % (et, self.current_iters + 1, self.end_iters, self.current_epochs + 1, self.get_current_lr(),
                                                      errG_percep.item() * self.alpha1,
                                                      errG_matching.item() * self.alpha2, errG_adv.item(),
                                                      errG_d.item() * self.gamma, errD.item(), errC.item())
                if self.train_with_cr:
                    infos += ' Loss_adv_view: %.4f' % errG_adv_view.item()

                print(infos)

                mylog = infos + '\n'
                self.logger(3, mylog)

                # 保存X'
                # fake = netG(xr[:4, :, :, :])
                if self.train_with_cr:
                    sample = torch.cat((xr[:4, :, :, :], fake_view[:4, :, :, :], fake[:4, :, :, :]), dim=3)
                else:
                    sample = torch.cat((xr[:4, :, :, :], fake[:4, :, :, :]), dim=3)
                vutils.save_image(sample,
                                  '%s/fake_samples_iters_%04d.png' % (self.out_samples, self.current_iters + 1),
                                  normalize=True)
            if (self.current_iters + 1) % self.model_save_step == 0:
                self.evaluation()
                self.save_models()
            if (self.current_iters + 1) % self.lr_update_step == 0 and self.current_iters + 1 >= self.iters_lr_decay:
                self.update_lr()
