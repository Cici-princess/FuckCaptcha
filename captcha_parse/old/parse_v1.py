from settings import *
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
EDGE     = 10


# img_path = 'sample/11213.jpg'
# img_path = 'sample/5669.jpg'
# # 50602 连字
# img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\sample_data\Type_1\0c47f352-4e7b-11ea-89b7-001a7dda7113.jpg'
# # 420 连字
# img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\sample_data\Type_1\0d03258a-4e7b-11ea-b59a-001a7dda7113.jpg'
# # 21721 数字难以识别
# img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\sample_data\Type_1\0bae515e-4e7b-11ea-9e7c-001a7dda7113.jpg'


IMG_DIR_NAME = "captcha_img_data"
IMG_DIR_FROM	= os.path.join(RESOURCE_DIR, IMG_DIR_NAME)
IMG_DIR_TO	    = os.path.join(RESOURCE_DIR, IMG_DIR_NAME + "_output")
os.makedirs(IMG_DIR_TO, exist_ok=True)


def show_imgs(imgs, stack="v", *args, gray=False, save_path=None, show=True, **kwargs):
	def _2gray(img):
		try:
			return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		except:
			return img
	def _2rgb(img):
		try:
			return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		except:
			return img
	cmap = "gray" if gray else None
	imgs = list(map(_2gray if cmap == "gray" else _2rgb, imgs))
	imgs = np.vstack(imgs) if stack == "v" else np.hstack(imgs)
	plt.figure(dpi=DPI)
	plt.axis("off")
	plt.imshow(imgs, cmap=cmap, *args, **kwargs)
	if save_path:
		plt.savefig(save_path)
	if show:
		plt.show()

def blur_and_otsu(img, blur_val=5):
	img = 255 - img
	img = cv2.GaussianBlur(img, (blur_val, blur_val), 0)
	ret, img2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return img2

def denoise(img):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, 2)
	img_draw = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
	rects = []
	for i, contour in enumerate(contours):
		if cv2.contourArea(contour) < 100:
			cv2.drawContours(img_draw, contours, i, (255, 0, 0), 1)
			cv2.drawContours(img, contours, i, 0, 2)
		else:
			# 绘制最小矩形
			# rect = cv2.minAreaRect(contour)
			# box  = np.int0(cv2.boxPoints(rect)) #  bottom left, top left, top right, bottom right
			# src_pts = box.astype("float32")
			# w, h = map(int, rect[1])
			# dst_pts = np.array([[0, h - 1],
			#                     [0, 0],
			#                     [w - 1, 0],
			#                     [w - 1, h - 1]], dtype="float32")
			# M = cv2.getPerspectiveTransform(src_pts, dst_pts)  # the perspective transformation matrix
			# rotated_img = cv2.warpPerspective(img, M, (w, h))  # directly warp the rotated rectangle to get the straightened rectangle
			# cv2.drawContours(img_draw, [box], 0, (0, 255, 0), 2)

			# 绘制标准矩形
			x, y, w, h = cv2.boundingRect(contour)
			if h < 20:
				cv2.drawContours(img_draw, contours, i, (255, 0, 0), 1)
				cv2.drawContours(img, contours, i, 0, 2)
			else:
				cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
				rects.append((x, y, w, h))
	rects.sort()
	return img, img_draw, rects

def run_otsu_test_dir(img_type, start =0, img_sum=100, batch_size =20, split=False):
	img_dir_from = os.path.join(IMG_DIR_FROM, "Type_{}".format(img_type))
	img_dir_to = os.path.join(IMG_DIR_TO, "TrainingType_{}_{}".format(img_type, time.time()))
	os.makedirs(img_dir_to, exist_ok=True)

	images_all = []
	for seq in range(start, img_sum, batch_size):
		logging.debug("Parsing [{}/{}]...".format(seq, img_sum))
		images_col = []
		for img_name in os.listdir(img_dir_from)[seq: seq + batch_size]:
			img_path = os.path.join(img_dir_from, img_name)
			img = cv2.imread(img_path)
			img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img_otsu = blur_and_otsu(img_gray)
			img_denoised, img_draw, rects = denoise(img_otsu)
			if split:
				for i, (x, y, w, h) in enumerate(rects):
					cv2.imwrite(
						"{}/{}_{}.png".format(img_dir_to, img_name.split(".")[0], i),
					     cv2.resize(img_denoised[max(y - EDGE, 0): min(y + h + EDGE, img_denoised.shape[0]),
					                max(x - EDGE, 0): min(x + w + EDGE, img_denoised.shape[1])], IMG_SPLIT_SIZE)
					)
			images_col.append(np.hstack([
				cv2.resize(img, IMG_SHOW_SIZE),
				cv2.resize(img_draw, IMG_SHOW_SIZE),
			]))
		images_all.append(np.vstack(images_col))

	# vision_path = os.path.join(IMG_DIR_TO, "gallery_type_{}_{}.png".format(img_type, time.time()))
	# logging.debug("Generating the full vision of type_{} at path: {}".format(img_type, vision_path))
	# show_imgs(images_all, gray=False, stack="h", show=False, save_path=vision_path)


run_otsu_test_dir(img_type=1, split=True, img_sum=1000, batch_size=100)

