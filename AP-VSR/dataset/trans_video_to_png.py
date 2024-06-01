# 将UCF101数据集中的视频提取为帧
from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
# import imgaug.augmenters as iaa
import cv2
import shutil

n_thread = 40

# file_name: v_ApplyEyeMakeup_g01_c01.avi
# class_path: xxx/Videos/ApplyEyeMakeup
# dst_class_path: xxx/jpgs/ApplyEyeMakeup
def vid2jpg(file_name, class_path, dst_dir_path):
    if '.avi' not in file_name:
        return
    name, ext = os.path.splitext(file_name)

    # name = name[:-3] + 'c01'
    dst_directory_path = os.path.join(dst_dir_path, name)  # dst_directory_path: xxx/jpgs/v_ApplyEyeMakeup_g01_c01

    video_file_path = os.path.join(class_path, file_name)  # video_file_path: xxx/Videos/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
    # try:
    if os.path.exists(dst_directory_path):
        # if not os.path.exists(os.path.join(dst_directory_path, '00000000.png')):
        #     subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
        #     print('remove {}'.format(dst_directory_path))
        #     os.mkdir(dst_directory_path)
        # else:
        print('*** convert has been done: {}'.format(dst_directory_path))
        return
    else:
        os.makedirs(dst_directory_path, exist_ok=True)
    # except:
    #     print(dst_directory_path)
    #     return
    cmd = 'ffmpeg -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/%08d.jpg\"'.format(video_file_path, dst_directory_path)
    # print(cmd)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def class_process(dir_path, dst_dir_path, class_name):  # dir_path: xxx/Videos dst_dir_path: xxx/jpgs ApplyEyeMakeup
    print('*' * 20, class_name, '*'*20)
    class_path = os.path.join(dir_path, class_name)  # class_path: xxx/Videos/ApplyEyeMakeup
    if not os.path.isdir(class_path):
        print('*** is not a dir {}'.format(class_path))
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)  # dst_class_path: xxx/jpgs/ApplyEyeMakeup
    if not os.path.exists(dst_class_path):
        os.makedirs(dst_class_path, exist_ok=True)

    vid_list = os.listdir(class_path)
    vid_list.sort()
    p = Pool(n_thread)
    from functools import partial
    worker = partial(vid2jpg, class_path=class_path, dst_dir_path=os.path.join(dst_dir_path, class_name))
    for _ in tqdm(p.imap_unordered(worker, vid_list), total=len(vid_list)):
        pass
    # p.map(worker, vid_list)
    p.close()
    p.join()

    print('\n')


def resize_to_lr(dir_path, dst_dir_path, dst_lq_path, class_name, cut_size=256):
    class_path = os.path.join(dir_path, class_name)
    img_list = os.listdir(class_path)
    img_list.sort()

    for img_name in img_list:
        print("Processing image_{}……".format(img_name))
        # img_name = img_name[:-3]
        img_path = os.path.join(class_path, img_name)

        if img_path.endswith('c01') or img_path.endswith('c02'):

            dst_class_path = os.path.join(dst_dir_path, class_name, img_name)
            dst_lq_class_path = os.path.join(dst_lq_path, class_name, img_name)
            if os.path.exists(dst_class_path):
                print('*** convert has been done: {}'.format(img_path))
                continue
            os.makedirs(dst_class_path, exist_ok=True)
            os.makedirs(dst_lq_class_path, exist_ok=True)

            # print("Deleting {}……".format(dst_class_path))
            # if os.path.exists(dst_class_path):
            #     shutil.rmtree(dst_class_path)
            # print("Deleting {}……".format(dst_lq_class_path))
            # if os.path.exists(dst_lq_class_path):
            #     shutil.rmtree(dst_lq_class_path)
            img_file_list = os.listdir(img_path)
            for img in img_file_list:
                # 原始图像读取
                image = cv2.imread(os.path.join(img_path, img))
                # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                # cv2.imshow("image", image)
                # cv2.waitKey(0)
                height, width = image.shape[0], image.shape[1]
                if height > width:
                    cropped_img = image[(height-width)//2 : (height+width)//2, 0:width]
                else:
                    cropped_img = image[0 : height, (width - height)//2 : (height+width)//2]

                # cv2.imshow("image", cropped_img)
                # cv2.waitKey(0)
                resized_img = cv2.resize(cropped_img, (cut_size, cut_size), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(dst_class_path, img), resized_img)
                # os.remove(os.path.join(dst_class_path, img))

                lq_img = cv2.resize(cropped_img, (cut_size//4, cut_size//4), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(dst_lq_class_path, img), lq_img)
                # os.remove(os.path.join(dst_lq_class_path, img))

if __name__ == "__main__":
    # dir_path = sys.argv[1]  # /home/xxx/data/projects/datasets/UCF101/Videos
    # dst_dir_path = sys.argv[2]  # /home/xxx/data/projects/datasets/UCF101/jpgs
    # dir_path = 'E:/LiuJia/Data/UCF101/test/'
    dir_path = 'D:/DataSets/UCF101/images/test/gt/'
    # dst_dir_path = 'D:/DataSets/UCF101/images/test/gt/'
    dst_dir_path = 'D:/DataSets/UCF101/images/train/gt_256/'
    dst_lq_path = 'D:/DataSets/UCF101/images/train/gt_64/'

    class_list = os.listdir(dir_path)
    class_list.sort()
    # for class_name in class_list:  # class_name: ApplyEyeMakeup
    #     class_process(dir_path, dst_dir_path, class_name)
    for class_name in class_list:  # class_name: ApplyEyeMakeup
        resize_to_lr(dir_path, dst_dir_path, dst_lq_path, class_name)