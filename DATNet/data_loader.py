import random

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
from glob import glob

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, transform, mode, atts):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.transform = transform
        self.mode = mode
        self.male_data = []
        self.fe_male_data = []
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.atts = atts
        if mode == 'test':
            self.preprocess_test()
            self.num_images = len(self.test_dataset)
        elif mode == 'evaluation':
            self.preprocess()
            self.num_images = len(self.test_dataset)
        else:
            self.preprocess()
            self.num_images = len(self.train_dataset)
            print(len(self.train_dataset))
            print('Finished preprocessing the CelebA dataset...')

    def preprocess_temp(self):
        files = glob(os.path.join(self.image_dir, '*.png'))
        for file in files:
            filename = file[-12:]
            label = [file[-5:-4] == '1']
            self.test_dataset.append([filename, label])
        print("the length of test_dataset: %d" % len(self.test_dataset))

    def preprocess_croped(self):
        lines = [line.rstrip().split() for line in open(self.attr_path, 'r')]
        files = glob(os.path.join(self.image_dir, '*.png'))
        for file in files:
            filename = file[-10:]
            index = int(filename[:6]) + 1
            label = [lines[index][21] == '1']
            self.test_dataset.append([filename, label])
        print("the length of test_dataset: %d" % len(self.test_dataset))

    def preprocess_test(self):
        """Preprocess the CelebA attribute file."""

        files_852 = glob(os.path.join(r'.\utils\real_852', '*.png'))
        files_852 = [x[-10:-4] for x in files_852]

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        lines = lines[200799:]

        idx = self.attr2idx[self.atts]
        num = 0
        for _, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []

            if self.atts == 'Male':
                if values[idx] != '1':
                    if num < 684:
                        num += 1
                    else:
                        continue
            else:
                if filename[:-4] not in files_852:
                    continue

            label.append(values[idx] == '1')
            self.test_dataset.append([filename, label])


    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        # random.seed(1234)
        # random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []

            for att in self.atts:
                label.append(values[self.attr2idx[att]] == '1')

            if i < 200799:
                self.train_dataset.append([filename, label])
            else:
                self.test_dataset.append([filename, label])

    def preprocess_bal(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]

        random.seed(1234)

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            idx = self.attr2idx['Male']
            idx_age = self.attr2idx['Young']
            label.append(values[idx] == '1')
            label.append(values[idx_age] == '1')
            if i < 200799:
                if values[idx] == '1':
                    self.male_data.append([filename, label])
                else:
                    self.fe_male_data.append([filename, label])
            else:
                self.test_dataset.append([filename, label])
        random.shuffle(self.fe_male_data)
        random.shuffle(self.male_data)

        # 平衡每个bach的男女比例，以保证loss稳定；适用于batchsize=12
        step = 8
        index = 0
        for _ in range(0, 10468):
            group = self.male_data[index:index+step] + self.fe_male_data[index:index+step]
            random.shuffle(group)
            self.train_dataset.extend(group)
            index += step

        print(len(self.train_dataset))
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        # print(filename)
        image = Image.open(os.path.join(self.image_dir, filename))
        filename_num = [int(filename[:6])]
        # return self.transform(image), filename, torch.FloatTensor(label)
        return self.transform(image), torch.FloatTensor(label), torch.IntTensor(filename_num)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, batch_size=16, dataset='CelebA', mode='train', num_workers=2, atts=('Male', 'Young')):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(178))
    transform.append(T.Resize(224))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, transform, mode, atts)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader
