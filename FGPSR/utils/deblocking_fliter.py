import cv2
import numpy as np


def calWeight(d, k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


# def imgFusion(img1, img2, overlap, left_right=True):
#     '''
#     图像加权融合
#     :param img1:
#     :param img2:
#     :param overlap: 重合长度
#     :param left_right: 是否是左右融合
#     :return:
#     '''
#     # 这里先暂时考虑平行向融合
#     w = calWeight(overlap, 0.05)  # k=5 这里是超参
#
#     if left_right:  # 左右融合
#         col, row = img1.shape
#         img_new = np.zeros((row, 2 * col - overlap))
#         img_new[:, :col] = img1
#         w_expand = np.tile(w, (col, 1))  # 权重扩增
#         img_new[:, col - overlap:col] = (1 - w_expand) * img1[:, col - overlap:col] + w_expand * img2[:, :overlap]
#         img_new[:, col:] = img2[:, overlap:]
#     else:  # 上下融合
#         row, col = img1.shape
#         img_new = np.zeros((2 * row - overlap, col))
#         img_new[:row, :] = img1
#         w = np.reshape(w, (overlap, 1))
#         w_expand = np.tile(w, (1, col))
#         img_new[row - overlap:row, :] = (1 - w_expand) * img1[row - overlap:row, :] + w_expand * img2[:overlap, :]
#         img_new[row:, :] = img2[overlap:, :]
#     return img_new

# def combine():
#     img1_ori = cv2.imread(r'../utils/data/left_SR_bicu2.png', cv2.IMREAD_UNCHANGED)
#     img2_ori = cv2.imread(r'../utils/data/right2.png', cv2.IMREAD_UNCHANGED)
#
#     img_out = np.zeros((img1_ori.shape[0], img1_ori.shape[1] + img1_ori.shape[1] - 16, img1_ori.shape[2]))
#
#     for i in range(3):
#         img_new = img_out[:][:][i]
#         img1 = img1_ori[:][:][:]
#         img2 = img2_ori[:][:][i]
#
#
#         img_new[:][:img1.shape[1] - 8] = img1[:][:img1.shape[1] - 8]
#         img_new[:][img1.shape[1] - 8:img1.shape[1]] = (img1[:][img1.shape[1]-8:img1.shape[1]] + img2[:][:8]) // 2
#         img_new[:][img1.shape[1]:][i] = img2[:][8:]
#
#         img_out[:][:][i] = img_new


def imgFusion(img1, img2, overlap, left_right=True):
    w = calWeight(overlap, 0.1)  # k=5 这里是超参
    row1, col1, chl1 = img1.shape
    row2, col2, chl2 = img2.shape
    img1_bak = img1
    img2_bak = img2
    if chl1 != chl2:
        print("图片拼接通道数不一致，退出")
        exit(-1)
    if left_right:
        img_new_dst = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1] - overlap, img1.shape[2]))
    else:
        img_new_dst = np.zeros((img1.shape[0] + img2.shape[0] - overlap, img1.shape[1], img1.shape[2]))
    for i in range(chl1):
        img1 = img1_bak[:, :, i]
        img2 = img2_bak[:, :, i]
        if left_right:  # 左右融合
            img_new = np.zeros((row1, col1 + col2 - overlap))
            img_new[0:row1, 0:col1] = img1
            w_expand = np.tile(w, (row1, 1))  # 权重扩增
            img_new[0:row1, (col1 - overlap):col1] = \
                (1 - w_expand) * img1[0:row1, (col1 - overlap):col1] + \
                w_expand * img2[0:row2, 0:overlap]
            img_new[:, col1:] = img2[:, overlap:]
            img_new_dst[:, :, i] = img_new
        else:  # 上下融合
            img_new = np.zeros((row1 + row2 - overlap, col1))
            img_new[0:row1, 0:col1] = img1
            w = np.reshape(w, (overlap, 1))
            w_expand = np.tile(w, (1, col1))
            img_new[row1 - overlap:row1, 0:col1] = \
                (1 - w_expand) * img1[(row1 - overlap):row1, 0:col1] + \
                w_expand * img2[0:overlap, 0:col2]
            img_new[row1:, :] = img2[overlap:, :]
            img_new_dst[:, :, i] = img_new
    return img_new_dst


if __name__ =="__main__":
    img1 = cv2.imread(r"./data/left_SR_bicu.png",cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(r"./data/right1.png",cv2.IMREAD_UNCHANGED)
    # img1 = (img1 - img1.min())/img1.ptp()
    # img2 = (img2 - img2.min())/img2.ptp()
    img_new = imgFusion(img1,img2,overlap=12,left_right=True)
    # img_new = np.uint16(img_new*65535)

    img_new = np.uint8(img_new)

    cv2.imwrite(r'./data/test_new2.png', img_new)

# def calWeight(d, k):
#     '''
#     :param d: 融合重叠部分直径
#     :param k: 融合计算权重参数
#     :return:
#     '''
#
#     x = np.arange(-d / 2, d / 2)
#     y = 1 / (1 + np.exp(-k * x))
#     return y
#
#
# def imgFusion(img1, img2, overlap, left_right=True):
#     '''
#         图像加权融合
#         :param img1:
#         :param img2:
#         :param overlap: 重合长度
#         :param left_right: 是否是左右融合
#         :return:
#     '''
#     # 这里先暂时考虑平行向融合
#     w = calWeight(overlap, 0.05)  # k=5 这里是超参
#     # #####单通道拼接代码
#     if left_right:  # 左右融合
#         row, col = img1.shape
#         img_new = np.zeros((row, 2 * col - overlap))
#
#         print("img1.shape:", img1.shape)
#         print("img_new.shape:", img_new.shape)
#         img_new[:, :col] = img1
#         w_expand = np.tile(w, (row, 1))  # 权重扩增
#         img_new[:, col - overlap:col] = (1 - w_expand) * img1[:, col - overlap:col] + w_expand * img2[:, :overlap]
#         img_new[:, col:] = img2[:, overlap:]
#     else:  # 上下融合
#         row, col = img1.shape
#         img_new = np.zeros((2 * row - overlap, col))
#         img_new[:row, :] = img1
#         w = np.reshape(w, (overlap, 1))
#         w_expand = np.tile(w, (1, col))
#         img_new[row - overlap:row, :] = (1 - w_expand) * img1[row - overlap:row, :] + w_expand * img2[:overlap, :]
#         img_new[row:, :] = img2[overlap:, :]
#     return img_new
#
#
# if __name__ == "__main__":
#     img1 = cv2.imread("pic/1.png", cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread("pic/2.png", cv2.IMREAD_UNCHANGED)
#
#     ##转单通道
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
#     img1 = (img1 - img1.min()) / img1.ptp()
#     img2 = (img2 - img2.min()) / img2.ptp()
#     img_new = imgFusion(img1, img2, overlap=128, left_right=True)
#
#     cv2.namedWindow("test_show", 0)
#     cv2.imshow("test_show", img_new)
#     cv2.waitKey(0)
#
#     img_new = np.uint16(img_new * 65535)
#     cv2.imwrite('pic/test_new.png', img_new)