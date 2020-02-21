# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/19 1:33
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import cv2
import numpy as np
from scipy import signal

def show_img(img, gray=True, title=""):
    if gray:
        plt.imshow(img, "gray")
    else:
        plt.imshow(img)
    plt.title(title)
    plt.show()


img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\quant\sample_data\Type_1\0a0c7b9c-4e7b-11ea-a0ad-001a7dda7113.jpg'

# 读取图片
im = cv2.imread(img_path)
show_img(im, gray=False, title="原图")

# 标准灰度化
im_1 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
show_img(im_1, title="灰度化")

# OSTU二值化
otsu_threshold, im_2 = cv2.threshold(im_1, 0, 1, cv2.THRESH_OTSU)
show_img(im_2, title="二值化")

## 反相
#im_2 = 1 - im_2
#show_img(im_2, title="反相")

# 卷积去除干扰线
im_3 = (signal.convolve2d(im_2, np.ones((3, 3)),
                          mode="same", boundary="symm") >= 6).astype(np.uint8)
show_img(im_3, title="去除干扰线")

# 二次卷积去除噪点
mask_scatter = signal.convolve2d(1-im_3, np.ones((10,10)), mode="same") == \
        signal.convolve2d(1-im_3, np.ones((20,20)), mode="same")
im_3[mask_scatter] = 1
show_img(im_3, title="去除离散点")


# 膨胀
im_4 = cv2.morphologyEx(im_3, cv2.MORPH_ERODE, np.ones((3, 3)))
show_img(im_4, title="膨胀运算")

# 去除纵向噪点
x_list = (1-im_4).sum(axis=0).astype(bool)
jj1 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
mask_x = signal.convolve(x_list, jj1,  mode="same") == 0
mask_x2 = np.logical_or(1-x_list, mask_x)
im_5 = im_4 * 255
im_5[:, mask_x2] = 200 # 将有噪点的变成白色
show_img(im_5, title="去除纵向噪点")

# 横向切割
img_stt_x_list = [i for i in range(1, 320) if mask_x2[i] < mask_x2[i-1]]
img_end_x_list = [i for i in range(1, 320) if mask_x2[i] > mask_x2[i-1]]
img_x_list = list(zip(img_stt_x_list, img_end_x_list))

def get_img_top(img: np.ndarray):
    for i, j in enumerate((1-img).sum(axis=1)):
        if j > 0:
            return i
        
def get_img_bottom(img: np.ndarray):
    for i, j in enumerate(reversed((1-img).sum(axis=1))):
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
    

imgs = []
for i, j in img_x_list:
    img_crop_x = im_5[:, i: j] / 255
    img_crop_xy = img_crop_x[get_img_top(img_crop_x): get_img_bottom(img_crop_x), :]
    img_scaled = scale_img(img_crop_xy)
    imgs.append(img_scaled)

show_img(np.hstack(imgs), title="标准化拼接")

