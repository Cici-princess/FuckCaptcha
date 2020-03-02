# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/21 21:41
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

import os
PROJECT_DIR     = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR    = os.path.join(PROJECT_DIR, "src")
MAPPING_DIR     = os.path.join(PROJECT_DIR, "mapping")
DATA_DIR        = os.path.join(PROJECT_DIR, "data")


import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)


import time
def calc_time(func):
	def wrapper(*args, **kwargs):
		st = time.time()
		f = func(*args, **kwargs)
		et = time.time()
		logging.debug("FUNC: {:15s}, TIME SPENT: {:.4f}".format(func.__name__, et-st))
		return f
	return wrapper


import cv2
def rot_img(img, angle):
	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	H, W = img.shape
	img = cv2.copyMakeBorder(img, *[int(H/2)]*4, cv2.BORDER_CONSTANT, 0)
	M = cv2.getRotationMatrix2D(tuple(map(lambda x: int(x/2), img.shape)), angle, 1)
	img = cv2.warpAffine(img, M, img.shape)
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, 2)
	# assert len(contours) == 1
	x, y, w, h = cv2.boundingRect(contours[0])
	img = img[y: y+h, x: x+w]
	img = cv2.copyMakeBorder(img, *[int(img.shape[0]/8)]*4, cv2.BORDER_CONSTANT, 0)
	img = cv2.resize(img, (H, W))
	_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	return img


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def show_img(img, cmap=None, hide_axis=False, block=False):
	plt.imshow(img, cmap=cmap)
	if hide_axis:
		plt.axis("off")
	if block:
		plt.ioff()
	plt.show()

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import pandas as pd

