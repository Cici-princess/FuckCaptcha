# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/1 0:33
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SHAPE   = (320, 160)
IMGS_GRID   = (200, 40)
ROOT_DIR = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data'

all_imgs = []
for home, dirs, files in os.walk(ROOT_DIR):
	if not dirs:
		for img_name in files:
			img_path = os.path.join(home, img_name)
			img = cv2.imread(img_path)
			img_ = cv2.resize(img, IMG_SHAPE)
			all_imgs.append(img_)
assert len(all_imgs) == 8000

all_imgs = np.reshape(np.vstack(all_imgs), (IMGS_GRID[0]*IMG_SHAPE[1], IMGS_GRID[1]*IMG_SHAPE[0], -1), order="F")
plt.figure(dpi=300); plt.imshow(all_imgs); plt.show()