# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import time
import warnings
from typing import Any

import cv2
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import model
from dataset import CUDAPrefetcher, PairedImageDataset
from imgproc import tensor_to_image
from utils import build_iqa_model, load_pretrained_state_dict, make_directory, AverageMeter, ProgressMeter, Summary


from ops.OmniSR import LR_SR,OmniSR,LR_SR_x8,LR_SR_INV

import numpy as np
from torchvision.models import resnet50
import torch
import tqdm

from time_cal import TimeRecorder


device = 'cuda:0'


def load_dataset(config: Any, device: torch.device) -> CUDAPrefetcher:
    test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                       config["TEST"]["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"])
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    test_test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_test_data_prefetcher


def build_model(config: Any, device: torch.device):
    g_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           growth_channels=config["MODEL"]["G"]["GROWTH_CHANNELS"],
                                                           num_rrdb=config["MODEL"]["G"]["NUM_RRDB"])
    g_model = g_model.to(device)

    # compile model
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model)

    return g_model


def build_model2(config: Any, device: torch.device):

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

    g_model=OmniSR(kwards=kwards).cuda()


    # compile model
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model)

    return g_model


def test1(
        g_model: nn.Module,
        test_data_prefetcher: CUDAPrefetcher,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        device: torch.device,
        config: Any,
) -> [float, float]:
    save_image = True
    save_image_dir = "./data/samples"

    if config["TEST"]["SAVE_IMAGE_DIR"]:
        save_image = True
        save_image_dir = os.path.join(config["SAVE_IMAGE_DIR"], config["EXP_NAME"])
        make_directory(save_image_dir)

    # set the model as validation model
    g_model.eval()

    with torch.no_grad():

        repetitions = 50
        dummy_input = torch.rand(1, 3, 64, 64).to(device)

        # Set the data set iterator pointer to 0 and load the first batch of data
        test_data_prefetcher.reset()
        batch_data = test_data_prefetcher.next()

        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
        # print('warm up ...\n')
        # with torch.no_grad():
        #     for _ in tqdm.tqdm(range(50)):
        #         _ = g_model(dummy_input)

        # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
        torch.cuda.synchronize()

        # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 初始化一个时间容器
        timings = np.zeros((repetitions, 1))

        print('testing ...\n')
        with torch.no_grad():
            for rep in tqdm.tqdm(range(repetitions)):
                starter.record()
                # start = time.time()
                _ = g_model(lr)
                ender.record()
                torch.cuda.synchronize()  # 等待GPU任务完成
                # end = time.time()
                # print(end - start)
                curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
                timings[rep] = curr_time

        avg = (timings.sum() - timings.max()) / (repetitions - 1)
        print('\navg={}\n'.format(avg))

def test(
        g_model: nn.Module,
        test_data_prefetcher: CUDAPrefetcher,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        device: torch.device,
        config: Any,
) -> [float, float]:
    save_image = True
    save_image_dir = "./data/samples"

    if config["TEST"]["SAVE_IMAGE_DIR"]:
        save_image = True
        save_image_dir = os.path.join(config["SAVE_IMAGE_DIR"], config["EXP_NAME"])
        make_directory(save_image_dir)

    # set the model as validation model
    g_model.eval()


    with torch.no_grad():

        repetitions = 50
        dummy_input = torch.rand(1, 3, 64, 64).to(device)

        TR = TimeRecorder(repetitions, True)

        # Set the data set iterator pointer to 0 and load the first batch of data
        test_data_prefetcher.reset()
        batch_data = test_data_prefetcher.next()

        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
        # print('warm up ...\n')
        # with torch.no_grad():
        #     for _ in tqdm.tqdm(range(50)):
        #         _ = g_model(dummy_input)

        # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
        torch.cuda.synchronize()


        print('testing ...\n')
        with torch.no_grad():
            for rep in tqdm.tqdm(range(repetitions)):
                TR.start()
                _ = g_model(lr)
                TR.end()


        avg = TR.avg_time()
        print('\navg={}\n'.format(avg))


def main() -> None:
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/test/ESRGAN_x4-DFO2K-Set5.yaml",
                        help="Path to test config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    device = torch.device("cuda", config["DEVICE_ID"])
    test_data_prefetcher = load_dataset(config, device)
    g_model = build_model(config, device)
    psnr_model, ssim_model = build_iqa_model(
        config["SCALE"],
        config["TEST"]["ONLY_TEST_Y_CHANNEL"],
        device,
    )

    # Load model weights
    g_model = load_pretrained_state_dict(g_model, config["MODEL"]["G"]["COMPILED"], config["MODEL_WEIGHTS_PATH"])
    # g_model.load_state_dict(torch.load('./SR_models/Omni4_194.pt'))


    test(g_model,
         test_data_prefetcher,
         psnr_model,
         ssim_model,
         device,
         config)


if __name__ == "__main__":
    main()
