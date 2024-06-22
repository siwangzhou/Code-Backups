import cv2
import os
from data_util import imresize_np

# if __name__ == '__main__':
#
#     imgs_name = os.listdir(r'G:\BaoXiu\EX\data\vedio_process\data\dance_raw\images')
#
#     for i, name in enumerate(imgs_name):
#         t = cv2.imread(os.path.join(r'G:\BaoXiu\EX\data\vedio_process\data\dance_raw\images1', name[1:]),
#                        cv2.IMREAD_UNCHANGED)
#
#         t = imresize_np(t, 1/4, True)
#         # t = cv2.resize(t, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
#
#         cv2.imwrite(r'G:\BaoXiu\EX\data\vedio_process\data\dance_raw\bic/%03d.png' % i, t)


if __name__ == '__main__':

    imgs_name = os.listdir(r'G:\BaoXiu\EX\data\vedio_process\data\dance\LRbicx4')

    for i, name in enumerate(imgs_name):
        gt = cv2.imread(os.path.join(r'G:\BaoXiu\EX\data\vedio_process\data\dance\LRbicx4', name), cv2.IMREAD_UNCHANGED)

        gt = imresize_np(gt, 4, True)
        # t = cv2.resize(t, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)

        # gt = gt[:1024, :1920, :]

        cv2.imwrite(os.path.join(r'G:\BaoXiu\EX\data\vedio_process\data\dance\sr_bic', name), gt)