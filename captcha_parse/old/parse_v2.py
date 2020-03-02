# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 6:55
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

from general import *
from pytorch.model.verify import *
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import cv2
import os
import time
import numpy as np
import pandas as pd


EDGE_EXTEND = 1 / 4
DPI             = 2000
IMG_SPLIT_SIZE  = (28, 28)
IMG_SHOW_SIZE   = (320, 160)


img_mapping_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\data\captcha_img_data_mapping.csv'
img_mapping = pd.read_csv(img_mapping_path)

def real_img_name(img_name: str):
	img_type, img_seq = img_name.split(".")[0].split("_")
	img_type = int(img_type) + 1
	return img_mapping["Type_{}".format(img_type)][int(img_seq)].split(".")[0]



def blur_and_otsu(img, blur_val=9):
	img = 255 - img
	img = cv2.GaussianBlur(img, (blur_val, blur_val), 0)
	ret, img2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return img2


def fast_verify_img(img, img_name):
	if "model" not in globals():
		global model
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = torch.load(MODEL_PATH).to(device)
		model.eval()  # 把模型转为test模式
	p_array = Variable(F.softmax(model(trans(img).to('cuda').unsqueeze(0)), dim=1)).cpu().numpy()
	img_info = {"img_name": img_name, "value": np.argmax(p_array), "prob": p_array.max()}
	logging.info(img_info)
	return str(img_info["value"])

def aft_process_split_img(img, interpolation=cv2.INTER_AREA):
	"""
	根据cv2官网介绍，缩小的时候使用cv2.INTER_AREA比较推荐
	[Geometric Transformations of Images — OpenCV-Python Tutorials 1 documentation](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html?highlight=resize )
	:param img:
	:param interpolation:
	:return:
	"""
	img = cv2.resize(img, IMG_SPLIT_SIZE, interpolation=interpolation)
	img = cv2.copyMakeBorder(img, *[int(img.shape[0] * EDGE_EXTEND)] * 4, cv2.BORDER_CONSTANT, value=0)
	img = cv2.resize(img, IMG_SPLIT_SIZE, interpolation=interpolation)
	_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	return img

def parse_img(img_path, show_img=False, verify=False):
	img = cv2.imread(img_path, 0)
	img2 = blur_and_otsu(img)

	contours, hierarchy = cv2.findContours(img2, cv2.RETR_EXTERNAL, 2)
	contours = list(sorted(contours, key=lambda x: min(i[0][0] for i in x)))
	split_imgs = []
	for i, contour in enumerate(contours):
		if cv2.contourArea(contour) < 100:
			cv2.drawContours(img2, contours, i, 0, 2)
		# 最小外接矩形
		else:
			rect = cv2.minAreaRect(contour)
			# print(rect)
			center, (w, h), angle = rect[0], rect[1], rect[2]
			center, w, h = tuple(map(int, center)), int(w), int(h)
			if angle > -45:
				rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
				img_rot = cv2.warpAffine(img2, rot_mat, (img2.shape[1], img2.shape[0]))
				img_crop = cv2.getRectSubPix(img_rot, (w, h), center)
			else:
				x, y, w, h = cv2.boundingRect(contour)
				img_crop = img2[max(y, 0): min(y+h, img2.shape[0]),
				           max(x, 0): min(x+w, img2.shape[1])]

			# 去除占用面积太大的
			if (img_crop/255).sum()/img_crop.size < 0.7:
				split_imgs.append(img_crop)

			if show_img:
				box = np.int0(cv2.boxPoints(rect))  # bottom left, top left, top right, bottom right
				cv2.drawContours(img2, [box], 0, 200, 2)
	if show_img:
		concat_img = np.vstack([
			cv2.resize(img2, IMG_SHOW_SIZE),
			cv2.resize(np.hstack([aft_process_split_img(img) for img in split_imgs]), IMG_SHOW_SIZE)])
		plt.imshow(concat_img, "gray")
		plt.show()
	return img, split_imgs

def parse_dir(img_dir_from, img_dir_to, verify=False, show_img=False, batch_size=10000, start=0,
              save=False, rotate=False, rotate_cnt=10, max_rotate=45):
	print("Started processing [{}/{}], batch_size {}".format(start, len(os.listdir(img_dir_from)), batch_size))
	all_imgs = []
	img_value_dict = dict()
	for img_name in os.listdir(img_dir_from)[start: start + batch_size]:
		img_path = os.path.join(img_dir_from, img_name)
		raw_img, split_imgs_raw = parse_img(img_path, show_img=False)
		split_imgs_aft = []
		for split_img_seq, split_img in enumerate(split_imgs_raw):
			img_name_pre, img_name_after = img_name, ""
			if "." in img_name:
				img_name_pre, img_name_after = img_name.split(".")
			split_img_name = "{}_{}".format(img_name_pre, split_img_seq)
			split_img_path = os.path.join(img_dir_to, split_img_name + "." + img_name_after)
			split_img = aft_process_split_img(split_img)    # ！需要先标准成正方形，否则后续旋转会有问题
			split_imgs_aft.append(split_img)

			if save:
				cv2.imwrite(split_img_path, split_img)

			if rotate:
				for angle in np.linspace(-max_rotate, max_rotate, rotate_cnt):
					angle = int(angle)
					img_rot = rot_img(split_img, angle)
					split_rot_img_name = "{}_{}".format(split_img_name, angle)
					split_rot_img_path = os.path.join(img_dir_to, split_rot_img_name + "." + img_name_after)
					cv2.imwrite(split_rot_img_path, img_rot)

		if verify:
			img_value_dict[img_name] = "".join(fast_verify_img(split_img, img_name) for split_img in split_imgs_aft)
		if show_img:
			all_imgs.append(np.hstack([
				cv2.resize(raw_img, IMG_SHOW_SIZE),
				cv2.resize(np.hstack([aft_process_split_img(img, interpolation=cv2.INTER_AREA) for img in split_imgs_aft]), IMG_SHOW_SIZE),
			]))

	if verify:
		df = pd.DataFrame([{"IMG_ID": real_img_name(i), "GUESS": j, "IMG_SEQ": i} for i, j in img_value_dict.items()])
		df.to_csv(VERIFIED_SAVE_PATH, encoding="utf_8_sig", index=False)
	if show_img:
		plt.imshow(np.vstack(all_imgs), "gray")
		plt.axis("off")
		plt.show()
	print("Finished processing images from {} into {}".format(img_type_from, img_type_to))
	return all_imgs


if __name__ == '__main__':

	# img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data\Type_1\2f027f40-57dc-11ea-9968-001a7dda7113.jpg'
	# img_value = parse_img(img_path)

	VERIFIED_SAVE_PATH   = 'results_type_1.csv'

	IMG_DIR_NAME = "captcha_img_data_renamed_v2"
	IMG_DIR_FROM	= os.path.join(RESOURCE_DIR, IMG_DIR_NAME)
	IMG_DIR_TO	    = os.path.join(RESOURCE_DIR, IMG_DIR_NAME + "_output")
	os.makedirs(IMG_DIR_TO, exist_ok=True)


	for img_type_dir_name in os.listdir(IMG_DIR_FROM):
		img_type_from = os.path.join(IMG_DIR_FROM, img_type_dir_name)
		img_type_to   = os.path.join(IMG_DIR_TO, img_type_dir_name)
		if os.path.isdir(img_type_from):
			os.makedirs(img_type_to, exist_ok=True)
			parse_dir(img_type_from, img_type_to,
		          start=0, batch_size=1000,
		          verify=False, save=True, rotate=False)


