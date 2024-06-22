'''
检测模型在LFW上的能力表现
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
sys.path.insert(0,'networks')
sys.path.insert(0,'lib')
import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.metrics as skm
from scipy import interpolate
from data_loader import get_loader
from utils.fmobilenet import FaceMobileNet
from torchvision import transforms as T
import numpy as np
import PIL

# configs
mean = (131.0912, 103.8827, 91.4953)

device = 'cuda:0'
tanh = nn.Tanh()

class Config:
    # network settings
    backbone = 'fmobile'  # [resnet, fmobile, sphere]
    metric = 'arcface'  # [cosface, arcface]
    embedding_size = 512
    drop_ratio = 0.5

    # data preprocess
    input_shape = [3, 112, 112]

    test_transform = T.Compose([
        T.CenterCrop(169), # crop 178 resize 224
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # dataset
    test_dir_origin = r'E:\BaoXiu\data\origin_imgs_crop178_resize224_1800'
    test_pairs = r'.\utils\CelebA_test.txt'

    # training settings
    restore = False
    restore_model = ""
    test_model = "./weights/matcher_for_evaluation/mobileNet_CosFace_e29.pth"

    train_batch_size = 160
    test_batch_size = 200

    epoch = 30
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 20
    lr_decay = 0.1
    weight_decay = 5e-4
    loss = 'cross_entropy'  # ['focal_loss', 'cross_entropy']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    pin_memory = True  # if memory is large, set it True to speed up a bit
    num_workers = 4  # dataloader

def face_matching(G=None, iters=0, plot=False, data_path=''):
    '''
    用LFW测试模型
    '''

    # 加载参数
    conf = Config()

    with torch.no_grad():

        # 获取测试对
        pairs = []
        with open(conf.test_pairs, 'r') as f:
            for line in f.readlines()[0:]:
                pair = line.strip().split()
                pairs.append(pair)
        pairs = np.array(pairs)

        # 获取图片路径
        paths, actual_issame = get_paths(os.path.expanduser(data_path), pairs, 'jpg')

        # model = ResIRSE(conf.embedding_size, conf.drop_ratio)
        # model = sphere(type=64, is_gray=None)
        model = FaceMobileNet(conf.embedding_size)
        model = model.to(conf.device)
        model.load_state_dict(torch.load(conf.test_model, map_location='cuda'))
        # model = senet50_ft(weights_path='model/senet50_ft_dims_2048.pth')
        # model = model.to(conf.device)

        # model = ResNet50()
        # load_state_dict(model, r'H:\liujia\SensativeNets\resnet50_scratch_weight.pkl')
        # model = model.to(conf.device)
        model.eval()


        # 前向传播计算特征
        # print('Runnning forward pass on LFW images')
        batch_size = conf.test_batch_size
        nrof_images = len(paths)
        nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array_origin = np.zeros((nrof_images, conf.embedding_size))             # 保存特征
        emb_array_p = np.zeros((nrof_images, conf.embedding_size))  # 保存特征

        #处理特征
        for i in range(nrof_batches):
            start_index = i * batch_size
            # print('Handing {}/{}'.format(start_index, nrof_images))
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]

            # images_origin = _preprocess(paths_batch, conf.test_transform).to(conf.device)
            # feats_origin = model(images_origin)
            # # 特征L2正则化
            # feats_origin = torch.nn.functional.normalize(feats_origin, p=2, dim=1)
            # emb_array_origin[start_index:end_index, :] = feats_origin.cpu().numpy()

            images_p = _preprocess(paths_batch, conf.test_transform, G).to(conf.device)
            feats_p = model(images_p)
            # 特征L2正则化
            feats_p = torch.nn.functional.normalize(feats_p, p=2, dim=1)
            emb_array_p[start_index:end_index, :] = feats_p.cpu().numpy()

        # 检测相关数据
        # score_0 = get_score(emb_array_origin)
        # scores_0 = np.array(score_0)

        score_1 = get_score(emb_array_p)
        scores_1 = np.array(score_1)




        # fpr_levels_0, tpr_at_fpr_0, roc_auc_0 = compute_ROC(actual_issame, scores_0)
        fpr_levels_1, tpr_at_fpr_1, roc_auc_1 = compute_ROC(actual_issame, scores_1)

        if plot:
            font1 = {'family': 'Times New Roman',
                     'weight': 'bold',
                     'size': 28,
                     }
            plt.figure(figsize=(25, 25))
            # plt.plot(fpr_levels_0, tpr_at_fpr_0, color='r',
            #          linestyle='--', marker='o', markersize=8,
            #          label='Original (Matching Rate = %0.6f)' % roc_auc_0)  ###假正率为横坐标，真正率为纵坐标做曲线
            plt.plot(fpr_levels_1, tpr_at_fpr_1, color='b',
                     linestyle='--', marker='^', markersize=8,
                     label='mutil-alter-epoch19 (Matching Rate = %0.6f)' % roc_auc_1)  ###假正率为横坐标，真正率为纵坐标做曲线



            # my_x_ticks = np.arange(0.0, 1, 0.1)
            # plt.xticks(my_x_ticks)
            # my_y_ticks = np.arange(0, 1.1, 0.1)
            # plt.yticks(my_y_ticks)
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver operating characteristic example')
            # plt.legend(loc="lower right")
            plt.xlabel('False Match Rate', font1)
            plt.ylabel('True Match Rate', font1)
            # plt.yticks(fontproperties='Times New Roman', size=28)
            # plt.xticks(fontproperties='Times New Roman', size=28)
            plt.yticks(fontproperties=font1)
            plt.xticks(fontproperties=font1)
            plt.title('Face Match ROC curves', font1)
            plt.legend(loc="lower right", prop=font1)

            plt.ylim(0, 1)
            # plt.xlim(0, 1)

            plt.savefig('epoch_%d_fm_%g.png' % (iters, roc_auc_1))
            # plt.show()
            plt.close()

        return roc_auc_1


def get_score(embeddings):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    from torch import cosine_similarity

    # print('==> compute template verification results.')

    score = []
    ## compute cosine_similarity of face1 & face2
    for i in range(embeddings1.shape[0]):
        inp1 = embeddings1[i]
        inp2 = embeddings2[i]
        face_inp1 =torch.Tensor(inp1)
        face_inp2 =torch.Tensor(inp2)
        # face_inp1 = face_inp1.view(face_inp1.shape[0], 1)
        # face_inp2 = face_inp2.view(face_inp2.shape[0], 1)

        similarity_score = cosine_similarity(face_inp1,face_inp2, dim=0)
        # similarity_score = face_inp1.mm(face_inp2.t())
        # print('similarity_score.shape:',similarity_score.size())
        similarity_scores = np.array(similarity_score)
        # print('similarity_score:',similarity_scores)
        score.append(similarity_scores)
    return score

def compute_ROC(labels, scores):
    # print('==> compute ROC.')
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = skm.roc_curve(labels, scores) ####计算真正率和假正率及阈值
    roc_auc = skm.auc(fpr, tpr)  ###计算auc的值,AUC为roc曲线下方的面积大小

    # fpr_levels = [ 5e-4, 1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2, 3e-2, 5e-2, 7e-2, 1e-1]
    fpr_levels = [ 5e-4, 1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7e-2, 9e-2, 1e-1, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84, 0.98]
    f_interp = interpolate.interp1d(fpr, tpr)
    tpr_at_fpr = [f_interp(x) for x in fpr_levels]
    return fpr_levels, tpr_at_fpr, roc_auc


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist


def _preprocess(images: list, transform, model=None) -> torch.Tensor:
    res = []

    crops = T.Compose([T.CenterCrop(178), T.Resize(224)])
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])

    for img in images:
        im = Image.open(img)
        im = crops(im)
        if model:
            im = to_tensor(im)
            im = im.view(1, 3, im.size(1), im.size(2))

            # fake = model(im.to(device))
            pertur = model(im.to(device))
            fake = tanh(pertur + im.to(device))

            im = fake / 2 + 0.5
            im = im.view(3, im.size(2), im.size(3))
            im = T.ToPILImage()(im)
        im = transform(im)
        im = torch.reshape(im, [1,3,112,112])
        res.append(im)
    data = torch.cat(res, dim=0)
    return data


def get_paths(lfw_dir, pairs, file_ext='jpg'):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        path0 = os.path.join(lfw_dir, pair[0][:-3] + file_ext)
        path1 = os.path.join(lfw_dir, pair[1][:-3] + file_ext)
        issame = (pair[4] == '1')
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

