"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
from os.path import exists
import shutil
from pathlib import Path


def get_train_test_lists(version='01'):
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    # Get our files based on version.
    test_file = os.path.join('E:/LiuJia/Data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist', 'testlist' + version + '.txt')
    # print('test_file路径：',test_file)
    train_file = os.path.join('E:/LiuJia/Data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist', 'trainlist' + version + '.txt')
    # print('train_file路径：',train_file)

    # Build the test list.
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]

    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]

    # Set the groups in a dictionary.
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups


def move_files(file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    for group, videos in file_groups.items():

        # Do each of our videos.
        for video in videos:

            # print('video:',video)
            # Get the parts.
            # parts = video.split(os.path.sep)
            parts = os.path.split(video)

            # print('不完整路径：',parts)
            classname = parts[0]
            # print('动作分类文件夹：'+classname)
            absolutepathname = 'E:/LiuJia/Data/UCF101/'
            # print('绝对路径：',absolutepathname)
            filename = parts[1]
            print('文件名:' + filename)
            # print('源代码中要产生的路径是：',os.path.join(group, filename))
            # Check if this class exists.
            '''
            if not os.path.exists(os.path.join(group, filename)):
                print("Creating folder for %s/%s" % (group, filename))
                os.makedirs(os.path.join(group, filename))
            '''

            # os.path <module 'ntpath' from 'D:\\python37\\lib\\ntpath.py'>

            if not exists(absolutepathname + group + '/' + classname):
                # print("新建文件夹:", (absolutepathname + group+ '/' + classname))
                os.makedirs(absolutepathname + group + '/' + classname)

            # Check if we have already moved this file, or at least that it
            # exists to move.
            file_to_move = absolutepathname + 'UCF-101/' + classname + '/' + filename
            if not os.path.exists(file_to_move):
                print("找不到要移动的文件.")
                continue

            # Move it.
            dest = Path(absolutepathname + group + '/' + classname + '/' + filename)
            print('目标路径:', dest)
            # os.rename(filename, dest)
            shutil.move(file_to_move, dest)
            print('move finished...')

    print("Done.")


def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    # Get the videos in groups so we can move them.
    group_lists = get_train_test_lists()
    print(group_lists)
    # Move the files.
    move_files(group_lists)


if __name__ == '__main__':
    main()

