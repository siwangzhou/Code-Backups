import os
import torch
import pyiqa
from tqdm import tqdm


class NIQE_metric(object):
    def __init__(self) -> None:
        device = torch.device("cuda")
        self.niqe_metric = pyiqa.create_metric('niqe', device=device)

    def __call__(self, img_path):
        ''' calculate NIQE value
        Args:
            img_path (str): image dir
        '''
        niqe_value = self.niqe_metric(img_path)  # crop_border=0
        # niqe_value = calculate_niqe(img, crop_border=0)
        return niqe_value.item()


photo_path = 'D:\code\Python\project\mechine Learning/block super resolution/test_img'
# photo_path = 'D:\code\Python\project\mechine Learning/block super resolution/test_img_GT'
test_list = os.listdir(photo_path)

cul_NIQE=NIQE_metric()
NIQE_sum=0
for imgname in tqdm(test_list):
    # for imgname in test_list:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    photo_str = photo_path + '/' + imgname

    NIQE=cul_NIQE(photo_str)
    NIQE_sum+=NIQE
    print(NIQE)
print('NIQE: ',NIQE_sum/len(test_list))