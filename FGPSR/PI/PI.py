import os
import torch
import pyiqa
from tqdm import tqdm


class NIQE_metric(object):
    def __init__(self) -> None:
        device = torch.device("cuda")
        self.niqe_metric = pyiqa.create_metric('pi', device=device)

    def __call__(self, img_path):
        ''' calculate NIQE value
        Args:
            img_path (str): image dir
        '''
        niqe_value = self.niqe_metric(img_path)  # crop_border=0
        # niqe_value = calculate_niqe(img, crop_border=0)
        return niqe_value.item()

dir_paths = [r'G:\BaoXiu\EX\ESRGAN-PyTorch-main\sr_imgs_4-28\temp']
# dir_paths = ['G:\BaoXiu\EX\ESRGAN-PyTorch-main\sr_imgs_4-28\ESRGAN', 'G:\BaoXiu\EX\ESRGAN-PyTorch-main\sr_imgs_4-28\Omni-SR']

for path in dir_paths:

    dir_list = os.listdir(path)

    for sub_path in dir_list:
        test_path = os.path.join(path, sub_path)

        test_list = os.listdir(test_path)

        cul_NIQE=NIQE_metric()
        NIQE_sum=0

        for imgname in tqdm(test_list):
            # for imgname in test_list:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            photo_str = test_path + '/' + imgname

            NIQE=cul_NIQE(photo_str)
            NIQE_sum+=NIQE
            # print(NIQE)
        print('%s PI:', (sub_path, NIQE_sum/len(test_list)))