# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/19 1:33
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

from general import *
from captcha_parse import *

import numpy as np
import cv2
from scipy import signal


def show_img(img, gray=True, title=""):
    if gray:
        plt.imshow(img, "gray")
    else:
        plt.imshow(img)
    plt.title(title)
    plt.show()

def get_img_top(img: np.ndarray):
    for i, j in enumerate((1 - img).sum(axis=1)):
        if j > 0:
            return i

def get_img_bottom(img: np.ndarray):
    for i, j in enumerate(reversed((1 - img).sum(axis=1))):
        if j > 0:
            return 160 - i

def scale_img(img: np.ndarray, obj_size=160):
    def multi_even(x):
        k = int(x * ratio)
        if k != 160 and k % 2 == 1:
            k += 1
        return k

    ratio = obj_size / max(img.shape)
    # cv2先输入宽，再输入高，顺序是反的
    img = cv2.resize(img, (multi_even(img.shape[1]), multi_even(img.shape[0])))
    padding = (obj_size - min(img.shape)) // 2
    if img.shape[0] > img.shape[1]:
        img = cv2.copyMakeBorder(img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=1)
    else:
        img = cv2.copyMakeBorder(img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=1)
    return img.astype(np.uint8)


def save_img(img_to_save, img_type, img_name, img_seq):
    img_save_dir = os.path.join(STANDARD_DATA_DIR, "Type_{}".format(img_type))
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    img_split_name = img_name.replace(".", "_{}.".format(img_seq))
    cv2.imwrite(os.path.join(img_save_dir, img_split_name), img_to_save)


def process_img(img_type, img_name):

    img_path = os.path.join(SAMPLE_DATA_DIR, "Type_{}/{}".format(img_type, img_name))
    # 读取图片
    im = cv2.imread(img_path)

    # 标准灰度化
    im_1 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    bright_degree = (im_1 / 255).mean()
    
    if bright_degree > 0.5:
        # OTSU二值化
        otsu_threshold, im_2 = cv2.threshold(im_1, 0, 1, cv2.THRESH_OTSU)
        # show_imgs(im_2, title="二值化")
        
    else:
        img3 = im_1 < 30
        img3[signal.convolve2d(img3, np.ones((5,5)), boundary="symm", mode="same") < 5] = False
        plt.imshow(img3, "gray")
        img4 = cv2.morphologyEx(img3.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5)))
        im_2 = 1 - img4

    # 卷积去除干扰线
    im_3 = (signal.convolve2d(im_2, np.ones((3, 3)),
                              mode="same", boundary="symm") >= 6).astype(np.uint8)
    # show_imgs(im_3, title="去除干扰线")

    # 二次卷积去除噪点
    mask_scatter = signal.convolve2d(1 - im_3, np.ones((10, 10)), mode="same") == \
                   signal.convolve2d(1 - im_3, np.ones((20, 20)), mode="same")
    im_3[mask_scatter] = 1
    # show_imgs(im_3, title="去除离散点")

    # 膨胀
    im_4 = cv2.morphologyEx(im_3, cv2.MORPH_ERODE, np.ones((3, 3)))
    # show_imgs(im_4, title="膨胀运算")

    # 去除纵向噪点
    x_list = (1 - im_4).sum(axis=0).astype(bool)
    jj1 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    mask_x = signal.convolve(x_list, jj1, mode="same") == 0
    mask_x2 = np.logical_or(1 - x_list, mask_x)
    im_5 = im_4 * 255
    im_5[:, mask_x2] = 200  # 将有噪点的变成白色
    # show_imgs(im_5, title="去除纵向噪点")

    save_img(im, img_type, img_name, "01")
    save_img(im_5, img_type, img_name, "02")

    # 横向切割
    img_stt_x_list = [i for i in range(1, 320) if mask_x2[i] < mask_x2[i-1]]
    img_end_x_list = [i for i in range(1, 320) if mask_x2[i] > mask_x2[i-1]]
    img_x_list = list(zip(img_stt_x_list, img_end_x_list))

    FINISHED = False
    while not FINISHED:
        for i, j in zip(img_x_list[:-1], img_x_list[1:]):
            if j[0] - i[1] < 10:
                img_x_list[i] = (img_x_list[i][0], img_x_list[j][1])
                img_x_list.pop(j)
                break
        else:
            FINISHED = True

    imgs = []
    for i, img_crop_tuple in enumerate(img_x_list):
        img_crop_x = im_5[:, img_crop_tuple[0]: img_crop_tuple[1]] / 255
        img_crop_xy = img_crop_x[get_img_top(img_crop_x): get_img_bottom(img_crop_x), :]
        if img_crop_xy.size > 0:
            img_scaled = scale_img(img_crop_xy)
            save_img(img_scaled*255, img_type, img_name, i+1)
            imgs.append(img_scaled)

    SHOW_IMG = False
    if SHOW_IMG:
        show_imgs(np.hstack(imgs), title="标准化拼接")


img_type = 1
for img_name in os.listdir(os.path.join(SAMPLE_DATA_DIR, "Type_{}".format(img_type))):
    print("Processing image {}".format(img_name))
    process_img(img_type, img_name)