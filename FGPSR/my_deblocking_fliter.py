from PIL import Image
import cv2
import numpy as np


# 引用H.265去除块效应

def vertical_block_fliter(img, x, top, bottom):
    for i in range(3):
        p0 = img[top:bottom, x - 1, i]
        p1 = img[top:bottom, x - 2, i]
        p2 = img[top:bottom, x - 3, i]
        p3 = img[top:bottom, x - 4, i]

        q0 = img[top:bottom:, x, i]
        q1 = img[top:bottom:, x + 1, i]
        q2 = img[top:bottom:, x + 2, i]
        q3 = img[top:bottom:, x + 3, i]

        img[top:bottom, x - 1, i] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) / 8.0
        img[top:bottom, x - 2, i] = (p2 + p1 + p0 + q0 + 2) / 4.0
        img[top:bottom, x - 3, i] = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) / 8.0

        img[top:bottom, x, i] = (q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) / 8.0
        img[top:bottom, x + 1, i] = (q2 + q1 + q0 + p0 + 2) / 4.0
        img[top:bottom, x + 2, i] = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) / 8.0


def horizontal_block_fliter(img, y, left, right):
    for i in range(3):
        p0 = img[y - 1, left:right, i]
        p1 = img[y - 2, left:right, i]
        p2 = img[y - 3, left:right, i]
        p3 = img[y - 4, left:right, i]

        q0 = img[y, left:right, i]
        q1 = img[y + 1, left:right, i]
        q2 = img[y + 2, left:right, i]
        q3 = img[y + 3, left:right, i]

        img[y - 1, left:right, i] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) / 8.0
        img[y - 2, left:right, i] = (p2 + p1 + p0 + q0 + 2) / 4.0
        img[y - 3, left:right, i] = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) / 8.0

        img[y, left:right, i] = (q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) / 8.0
        img[y + 1, left:right, i] = (q2 + q1 + q0 + p0 + 2) / 4.0
        img[y + 2, left:right, i] = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) / 8.0


def deblocking_h265(img, box):
    left, top, right, bottom = box

    img = np.float64(img)

    # 处理左右边界
    vertical_block_fliter(img, left, top, bottom)
    vertical_block_fliter(img, right, top, bottom)

    # 处理上下边界
    horizontal_block_fliter(img, top, left, right)
    horizontal_block_fliter(img, bottom, left, right)

    img = np.uint8(np.clip(img, 0, 255))

    return img


# 加权融合法去除块效应 引入overlap

def calWeight(d, k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def deblocking_weight_fusion(img, obj, box, overlap):
    box = [4 * x for x in box]

    left, top, right, bottom = box

    img = np.float64(img)
    obj = np.float64(obj)

    height, width, _ = obj.shape

    w = calWeight(overlap, 0.05)  # k是超参

    img_new = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for i in range(3):
        w_expand = np.tile(w, (height, 1))

        # left -- right
        img_new[top - overlap:bottom + overlap, left - overlap:left, i] = (1 - w_expand) * img[
                                                                                       top - overlap:bottom + overlap,
                                                                                       left - overlap:left,
                                                                                       i] + w_expand * obj[
                                                                                                       :,
                                                                                                       :overlap,
                                                                                                       i]
        # right --left
        img_new[top - overlap:bottom + overlap, right:right + overlap, i] = w_expand * img[top - overlap:bottom + overlap,
                                                                                   right:right + overlap, i] + (
                                                                                1 - w_expand) * obj[
                                                                                                :,
                                                                                                -overlap:, i]

        # w_expand = np.tile(w, (width - 2 * overlap, 1)).T
        w_expand = np.tile(w, (width, 1)).T

        # # top -- bottom
        # img[top - overlap:top, left:right, i] = (1 - w_expand) * img[top - overlap:top, left:right, i] + \
        #                                         w_expand * obj[:overlap, overlap:-overlap, i]
        # # bottom -- top
        # img[bottom:bottom + overlap, left:right, i] = w_expand * img[bottom:bottom + overlap, left:right, i] + \
        #                                               (1 - w_expand) * obj[-overlap:, overlap:-overlap, i]

        # top -- bottom
        img_new[top - overlap:top, left-overlap:right+overlap, i] = (1 - w_expand) * img[top - overlap:top, left-overlap:right+overlap, i] + \
                                                w_expand * obj[:overlap, :, i]
        # bottom -- top
        img_new[bottom:bottom + overlap, left-overlap:right+overlap, i] = w_expand * img[bottom:bottom + overlap, left-overlap:right+overlap, i] + \
                                                      (1 - w_expand) * obj[-overlap:, :, i]

        # middle block
        img_new[top:bottom:, left:right, i] = obj[overlap:-overlap, overlap:-overlap, i]

        # out block
        img[top-overlap:bottom+overlap:, left-overlap:right+overlap, i] = img_new[top-overlap:bottom+overlap:, left-overlap:right+overlap, i]


    img = np.uint8(img)

    # cv2.imshow("test", img_new)
    # cv2.waitKey(0)

    # cv2.imwrite(r'../utils/data/combine3.png', img_new)
    return img


if __name__ == "__main__":
    img = cv2.imread(r'./data/SRtttttt.png', cv2.IMREAD_UNCHANGED)

    img = deblocking_h265(img, (180, 132, 436, 260))

    # img = cv2.imread(r'./data/Trad_SR.png', cv2.IMREAD_UNCHANGED)
    # obj = cv2.imread(r'./data/samples/right.png', cv2.IMREAD_UNCHANGED)
    #
    # img = deblocking_weight_fusion(img, obj, (45, 33, 109, 65), 8)

    cv2.imwrite(r'./data/SRtttttt_deblocking1.png', img)
