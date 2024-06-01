import torch
import torchvision.transforms as T
import numpy as np
import cv2

def resize_sequences(sequences,target_size):
    """resize sequence
    Args:
        sequences (Tensor): input sequence with shape (n, t, c, h, w)
        target_size (tuple): the size of output sequence with shape (H, W)
    Returns:
        Tensor: Output sequences with shape (n, t, c, H, W)
    """
    seq_list=[]
    for sequence in sequences:
        img_list=[T.Resize(target_size,interpolation=T.InterpolationMode.BICUBIC)(lq_image) for lq_image in sequence]
        seq_list.append(torch.stack(img_list))
    
    return torch.stack(seq_list)

def rgb2normalizedLab(img_):
    # Convert to bgr format
    img = img_[:, :, ::-1]

    # Convert to L*a*b* format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Normalize the lab image to [0, 1] range
    img[:, :, 0] = img[:, :, 0] / 100.
    img[:, :, 1] = (img[:, :, 1] + 128.) / 255.
    img[:, :, 2] = (img[:, :, 2] + 128.) / 255.

    return img

def normalizedLab2rgb(img_):
    # Scale normlab image to original L*a*b* range
    img = img_.copy()
    img[:, :, 0] = img[:, :, 0] * 100.
    img[:, :, 1] = (img[:, :, 1] * 255.) - 128.
    img[:, :, 2] = (img[:, :, 2] * 255.) - 128.

    # Convert to rgb format
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

def tensor_rgb2lab(rgb):
    T, C, H, W = rgb.shape

    # Convert to L*a*b* space
    lab = []
    for t in range(T):
        frm_rgb = np.transpose(rgb[t, :, :, :].numpy(), (1, 2, 0)) # [H, W, C]
        frm_lab = rgb2normalizedLab(frm_rgb)
        frm_lab = np.transpose(frm_lab, (2, 0, 1)) # [C, H, W]
        lab.append(torch.from_numpy(frm_lab))
    return torch.stack(lab, dim=0)

def tensor_lab2rgb(out):
    out = out.clone().numpy()
    out_rgb = np.zeros(out.shape)
    for i, sample in enumerate(out):
        for j, frm_lab in enumerate(sample):
            frm_lab = np.transpose(frm_lab, (1, 2, 0)) # channels last
            frm_rgb = normalizedLab2rgb(frm_lab)
            frm_rgb = np.transpose(frm_rgb, (2, 0, 1)) # channels first
            out_rgb[i, j, :, :, :] = frm_rgb
    return torch.from_numpy(out_rgb)