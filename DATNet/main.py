import os
import argparse
from solver import DATNet_solver
from data_loader import get_loader
from torch.backends import cudnn
import sys


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # New trian or resume
    if config.is_continue == 0:
        if config.start_iters != 0:
            print('start_epoch not eql 0')
            sys.exit(1)
        if config.g_weights_path != '':
            print('wrong2!')
            config.netG = ''
            config.netD = ''
    else:
        config.g_weights_path = './out/model/G/netG_iters_%06d.pth' % config.start_epoch
        config.d_weights_path = './out/model/D/netD_iters_%06d.pth' % config.start_epoch
        config.c_weights_path = './out/model/C/netC_iters_%06d.pth' % config.start_epoch

    # Create directories if not exist.
    if not os.path.exists(config.out_log):
        os.makedirs(config.out_log)
    for model in ['G', 'D', 'C']:
        if not os.path.exists(config.out_model + '/' +model):
            os.makedirs(config.out_model + '/' + model)
    if not os.path.exists(config.out_fake):
        os.makedirs(config.out_fake)

    # Data loader.

    celeba_loader = get_loader(config.data_address, config.attr_address, config.batch_size,
                                  'CelebA', 'train', config.workers, config.attrs)
    

    # Solver for training and testing StarGAN.
    DATNet = DATNet_solver(celeba_loader, config)

    if config.mode == 'train':
        DATNet.train()
    elif config.mode == 'test':
        print('test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help="the mode of DATNet")
    parser.add_argument('--train_with_cr', default=False, help="train with cr?")
    parser.add_argument('--gan_mode', default='vanilla', help="gan_mode in ['vanilla', 'lsgan', 'wgangp']")
    parser.add_argument('--g_normalization', default='IN', help="gan_mode in ['IN', 'BN']")
    parser.add_argument('--d_normalization', default='no_N', help="gan_mode in ['IN', 'no_N']")
    parser.add_argument('--annotations', default='AT + TF + G_IN + D_IN + D(patchGAN) + ( 1 - b) * Ltf', help="annotations")


    parser.add_argument('--device', default='cuda:0', help="select a  device to train")
    parser.add_argument('--out', default='./out', help='folder to output images and model checkpoints')
    parser.add_argument('--out_fake', default='./out/samples', help='folder to output images and model checkpoints')
    parser.add_argument('--out_model', default='./out/model', help='folder to output G')
    parser.add_argument('--out_log', default='./out/log', help='folder to output log')
    parser.add_argument('--data_address', default=r'E:\BaoXiu\data\datas\dd\img_align_celeba')
    parser.add_argument('--attr_address', default=r'E:\BaoXiu\data\datas\Anno\list_attr_celeba.txt')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=12, help='input batch size')
    parser.add_argument('--end_iters', type=int, default=200000, help='iters of training')  # as 6 epoch
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--iters_lr_decay', type=float, default=50000, help='iters of begining decaying lr')

    parser.add_argument('--attrs', default=('Male', 'Young'), help='male or female or all')

    parser.add_argument('--alpha1', type=float, default=4, help='the rate of errG_percep')
    parser.add_argument('--alpha2', type=float, default=1, help='the rate of errG_matching')
    parser.add_argument('--beta_gender', type=float, default=3, help='the rate of errGender')
    parser.add_argument('--beta_age', type=float, default=4, help='the rate of errAge')
    parser.add_argument('--gamma', type=float, default=1, help='the rate of errGd')
    parser.add_argument('--a', default=0.5)
    parser.add_argument('--b', default=1.3)

    parser.add_argument('--n_g', default=1)
    parser.add_argument('--n_c', default=30)
    parser.add_argument('--log_save_step', default=1000)
    parser.add_argument('--model_save_step', default=2000)
    parser.add_argument('--lr_update_step', default=2000)


    parser.add_argument('--is_continue', type=int, default=0, help='resume?')
    parser.add_argument('--start_iters', type=int, default=0, help='the iters which has been trained')
    parser.add_argument('--g_weights_path', default=r'', help="path to netG (to continue training)")
    parser.add_argument('--d_weights_path', default=r'', help="path to netD (to continue training)")
    parser.add_argument('--c_weights_path', default=r'./weights/auxiliary_classifier_pretrained_weights/EfficientB0_pretrained_0.9911_e16_gender_age.pth',
                        help="path to netC (to continue training)")

    config = parser.parse_args()
    print(config)
    main(config)