# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 6:55
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

from settings import *
from pytorch.model.verify import *
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import cv2
import os
import time
import numpy as np


DPI             = 2000
IMG_SHOW_SIZE   = (320, 160)
IMG_SPLIT_SIZE  = (28, 28)
EDGE     = 5


IMG_DIR_NAME = "captcha_img_data"
IMG_DIR_FROM	= os.path.join(RESOURCE_DIR, IMG_DIR_NAME)
IMG_DIR_TO	    = os.path.join(RESOURCE_DIR, IMG_DIR_NAME + "_output")
os.makedirs(IMG_DIR_TO, exist_ok=True)


def blur_and_otsu(img, blur_val=9):
	img = 255 - img
	img = cv2.GaussianBlur(img, (blur_val, blur_val), 0)
	ret, img2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return img2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_PATH).to(device)
model.eval()    # 把模型转为test模式

def fast_verify_img(img, img_name):
	p_array = Variable(F.softmax(model(trans(img).to('cuda').unsqueeze(0)), dim=1)).cpu().numpy()
	img_info = {"img_name": img_name, "value": np.argmax(p_array), "prob": p_array.max()}
	logging.info(img_info)
	return str(img_info["value"])

def parse_img(img_path, show_img=False):
	img_name = img_path.split("/")[-1].split("\\")[-1]
	img = cv2.imread(img_path, 0)
	img = blur_and_otsu(img)

	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, 2)
	contours = list(sorted(contours, key=lambda x: min(i[0][0] for i in x)))
	split_imgs = []
	img_value = ""
	for i, contour in enumerate(contours):
		if cv2.contourArea(contour) < 100:
			cv2.drawContours(img, contours, i, 0, 2)
		# 最小外接矩形
		else:
			rect = cv2.minAreaRect(contour)
			# print(rect)
			center, (w, h), angle = rect[0], rect[1], rect[2]
			center, w, h = tuple(map(int, center)), int(w), int(h)
			if angle > -45:
				rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
				img_rot = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
				img_crop = cv2.getRectSubPix(img_rot, (w+EDGE*2, h+EDGE*2), center)
			else:
				x, y, w, h = cv2.boundingRect(contour)
				img_crop = img[max(y-EDGE, 0): min(y+h+EDGE, img.shape[0]),
				           max(x-EDGE, 0): min(x+w+EDGE, img.shape[1])]
			standard_img = cv2.resize(img_crop, IMG_SPLIT_SIZE)
			_, standard_img = cv2.threshold(standard_img, 127, 255, cv2.THRESH_BINARY)
			split_imgs.append(standard_img)

			img_value += fast_verify_img(standard_img, img_name)
			if show_img:
				box = np.int0(cv2.boxPoints(rect))  # bottom left, top left, top right, bottom right
				cv2.drawContours(img, [box], 0, 200, 2)
	if show_img:
		concat_img = np.vstack([cv2.resize(img, IMG_SHOW_SIZE), cv2.resize(np.hstack(split_imgs), IMG_SHOW_SIZE)])
		plt.imshow(concat_img, "gray")
		plt.show()
	return img_value

def parse_dir(img_dir):
	img_value_dict = dict()
	for img_name in os.listdir(img_dir):
		img_path = os.path.join(img_dir, img_name)
		img_value = parse_img(img_path, show_img=False)
		img_value_dict[img_name] = img_value

	import pandas as pd
	df = pd.DataFrame([{"IMG_ID": i.replace(".jpg", ""), "GUESS": j} for i, j in img_value_dict.items()])
	df.to_csv("type_1.csv", encoding="utf_8_sig", index=False)
	return img_value_dict



# img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data\Type_1\2f027f40-57dc-11ea-9968-001a7dda7113.jpg'
# img_value = parse_img(img_path)

img_dir = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data\Type_1'
img_value_dict = parse_dir(img_dir)