from data_loader import get_loader
import torch.nn.functional as F
import argparse
import torch
from utils.senet50.se_resnet import se_resnet50

parser = argparse.ArgumentParser()
#################### 参数 ########################################################################
parser.add_argument('--data_path', default=r'E:\BaoXiu\data\datas\dd\img_align_celeba')
##################################################################################################
parser.add_argument('--attr_path', default=r'E:\BaoXiu\data\datas\Anno\list_attr_celeba.txt')
opt = parser.parse_args()


def attrs_classification(G, att='gender', data_path='', attr_path='', device='cuda'):
    if att == 'gender':
        classifier = se_resnet50(r'./weights/classifiers_for_evaluation/Senet50_0.9900_e28.pth', num_classes=1)
        # dataset for gender classification
        data_loader = get_loader(data_path, attr_path, 1, 'CelebA', 'test', 0, 'Male')
    else:
        classifier = se_resnet50(r'./weights/classifiers_for_evaluation/senet50_e6_0.882629.pth', num_classes=1)
        # dataset for age classification
        data_loader = get_loader(data_path, attr_path, 1, 'CelebA', 'test', 0, 'Young')

    classifier.to('cuda')
    classifier.eval()

    num = 0
    acc = 0
    for i, (x, y, _) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        perturbation = G(x)
        fake = F.tanh(perturbation + x)

        predict = torch.sigmoid(classifier(fake))
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        acc += torch.eq(predict, y).sum().item()
        num += x.shape[0]

    acc = acc / num
    return acc
