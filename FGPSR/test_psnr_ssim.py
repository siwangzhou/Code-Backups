import cv2
from skimage.metrics import structural_similarity as compare_ssim
import os
from data_util import imresize_np
import numpy as np
import tqdm


if __name__ == "__main__":

    lr_path = r'G:\BaoXiu\EX\ESRGAN-PyTorch-main\sr_imgs_4-28\ESRGAN\det_ESRGAN_deblock'
    hr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\GT_crop'

    lr_name = os.listdir(lr_path)
    lr_name.sort(key=lambda x: int(x[:-4]))

    psnr = 0.0
    ssim = 0.0

    num = 0

    for name in tqdm.tqdm(lr_name):

        img = cv2.imread(os.path.join(lr_path, name))
        img_ori = cv2.imread(os.path.join(hr_path, name))

        img = img[:1024, :, :]

        # img_ori = np.float32(img_ori)
        # img = np.float32(img)
        #
        # img = imresize_np(img_ori, 1/4, True)
        # img = imresize_np(img, 4, True)

        # img = np.uint8(img)


        psnr += cv2.PSNR(img_ori, img)

        # 将图像转换为灰度，因为SSIM对颜色通道是分别计算的
        grayA = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算两个灰度图像的SSIM
        ssim_value, _ = compare_ssim(grayA, grayB, full=True)

        ssim += ssim_value

        num += 1

    print("psnr : %g,  ssim : %g" % (psnr / float(num), ssim / float(num)))

# if __name__ == "__main__":
#     gt_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\GT'
#     sr_path = r'G:\BaoXiu\EX\data\vedio_process\data\dance\sr_bic'
#
#
#     names = os.listdir(gt_path)
#
#     psnr = 0.0
#     num = 0
#     for name in names:
#         gt = cv2.imread(os.path.join(gt_path, name))
#         sr = cv2.imread(os.path.join(sr_path, name))
#
#         psnr += cv2.PSNR(gt, sr)
#         num += 1
#
#     print('psnr: %g' % (psnr/num))