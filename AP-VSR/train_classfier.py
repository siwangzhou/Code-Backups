import os

import torch
import timeit
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset.dataset import UCF101Dataset
from models.c3d_network import C3D
from models.r3d_model import R3DClassifier
from models.R2Plus1D_model import R2Plus1DClassifier
from utils import resize_sequences
from loss import FocalLoss

from torch.cuda.amp import autocast

def train_model():
    # dataloader
    gt_dir = 'D:/DataSets/UCF101/images/train/gt_256/'
    lq_dir = 'D:/DataSets/UCF101/images/train/gt_64/'
    label_path = "E:/LiuJia/Data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
    ucf101_dataset = UCF101Dataset(gt_dir, lq_dir, label_path, patch_size=64)
    train_loader = DataLoader(ucf101_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    test_gt_dir = 'D:/DataSets/UCF101/images/test/gt_256/'
    test_lq_dir = 'D:/DataSets/UCF101/images/test/gt_64/'
    test_ucf101_dataset = UCF101Dataset(test_gt_dir, test_lq_dir, label_path, patch_size=64)
    test_loader = DataLoader(test_ucf101_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Use GPU if available else revert to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    # train set
    resume_epoch = 0
    num_epochs = 60
    snapshot = 5
    max_test_acc = 0.0
    total_steps = num_epochs * (len(ucf101_dataset) // 4)
    save_dir = 'E:/LiuJia/VPSR/log/20240418/'
    os.makedirs(save_dir, exist_ok=True)
    lr = 1e-3
    model_name = 'R2Plus1D'
    num_classes = 101
    # model = C3D(num_classes=num_classes)
    # model = R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
    model = R2Plus1DClassifier(101, (2, 2, 2, 2), pretrained=False)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr,
    #                                       betas=(0.5, 0.999))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #             optimizer, lambda step: 1 - step / total_steps)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    # start training
    print("Training {} from scratch...".format(model_name))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    for epoch in range(resume_epoch, num_epochs):
        start_time = timeit.default_timer()

        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

        # scheduler.step() is to be called once every epoch during training
        scheduler.step()
        model.train()

        loop = tqdm(train_loader)
        for gts, lqs, labels in loop:
            gts = gts.transpose(1, 2)
            inputs = Variable(gts, requires_grad=True).to(device)
            labels = torch.LongTensor(labels)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
            loss.backward()
            optimizer.step()

            running_corrects = torch.sum(preds == labels.data)

            # 更新训练信息
            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=loss.data, acc=running_corrects.data/len(labels), lr=optimizer.state_dict()['param_groups'][0]['lr'])

        if (epoch + 1) % snapshot == 0:
            with torch.no_grad():
                model.eval()
                print("Testing {} from scratch...".format(model_name))

                num_correct = 0.0
                num_total = 0.0

                loop = tqdm(test_loader)
                for gts, lqs, labels in loop:
                    gts = gts.transpose(1, 2)
                    inputs = Variable(gts, requires_grad=True).to(device)
                    labels = torch.LongTensor(labels)
                    labels = Variable(labels).to(device)
                    with autocast():
                        outputs = model(inputs)
                        preds = torch.max(outputs, 1)[1]
                    num_total += len(labels)
                    num_correct += torch.sum(preds == labels.data)

                    loop.set_postfix(acc=num_correct / num_total)

                print("Testing is done.")
                print("Acc of test in {} is {:.4f}.".format(model_name, num_correct / num_total))
                now_test_acc = num_correct / num_total

                if now_test_acc > max_test_acc:
                    max_test_acc = now_test_acc
                    torch.save(model, os.path.join(save_dir, '{}_best.pth'.format(model_name)))



    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

def test_model():
    gt_dir = 'D:/DataSets/UCF101/images/test/gt_256/'
    lq_dir = 'D:/DataSets/UCF101/images/test/gt_64/'
    label_path = "E:/LiuJia/Data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
    ucf101_dataset = UCF101Dataset(gt_dir, lq_dir, label_path, patch_size=64)
    test_loader = DataLoader(ucf101_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    model_name = 'C3D'
    num_classes = 101
    ckpt_path = 'E:/LiuJia/VPSR/log/20240327/C3D_best.pth'
    # ckpt_path = 'E:/LiuJia/VPSR/log/20240417/R3D_best.pth'
    # ckpt_path = 'E:/LiuJia/VPSR/log/20240418/R2Plus1D_best.pth'

    if model_name == 'R3D':
        model = R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
    if model_name == 'C3D':
        model = C3D(num_classes=num_classes)
    else:
        model = R2Plus1DClassifier(101, (2, 2, 2, 2), pretrained=False)
    model = model.to(device)
    # state_dict = torch.load(ckpt_path)
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(ckpt_path).items()})
    # model.load_state_dict(torch.load(ckpt_path))
    model = torch.load(ckpt_path)

    print("Testing {} from scratch...".format(model_name))

    num_correct = 0.0
    num_total = 0.0
    with torch.no_grad():
        model.eval()
        loop = tqdm(test_loader)
        for gts, lqs, labels in loop:
            labels = torch.LongTensor(labels)
            labels = Variable(labels).to(device)

            # lqs = resize_sequences(lqs, (gts.size(dim=3), gts.size(dim=4)))
            inputs = Variable(gts, requires_grad=True).to(device)
            inputs = inputs.transpose(1, 2)

            # with autocast():
            outputs = model(inputs)
            preds = torch.max(outputs, 1)[1]
            num_total += len(labels)
            num_correct += torch.sum(preds == labels.data)

            loop.set_postfix(acc=num_correct/num_total)

    print("Testing is done.")
    print("Acc of test in {} is {:.4f}.".format(model_name, num_correct/num_total))



if __name__ == '__main__':
    # train_model()
    test_model()