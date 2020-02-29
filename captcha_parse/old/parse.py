# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 0:16
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------



def run_by_kMeans(img_path):
	img = cv2.imread(img_path)
	Z = img.reshape((-1,3))
	# convert to np.float32
	Z = np.float32(Z)
	K = 2
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))

	cv2.imshow('res2',res2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def run_by_adaptive(img_path):
	img = cv2.imread(img_path,0)
	img = cv2.medianBlur(img,5)

	ret,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

	titles = ['Original Image', 'Global Thresholding (v = 127)',
				'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [img, th1, th2, th3]

	for i in range(4):
		plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.show()

def run_by_otsu_tutorial(img_path):
	img = cv2.imread(img_path,0)

	# global thresholding
	ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

	# Otsu's thresholding
	ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Otsu's thresholding after Gaussian filtering
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# plot all the images and their histograms
	images = [
		(img, th1),
		(img, th2),
		(blur, th3)
	]
	titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
			  'Original Noisy Image','Histogram',"Otsu's Thresholding",
			  'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

	for i, (img_bf, img_af) in enumerate(images):
		plt.subplot(3,3,i*3+1),plt.imshow(img_bf,'gray')
		plt.title(titles[i*3]), plt.axis("off")
		plt.subplot(3,3,i*3+2), plt.hist(img_bf.ravel(),256)
		plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
		plt.subplot(3,3,i*3+3),plt.imshow(img_af,'gray')
		plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
	plt.show()

def test_one_img(img_path):
	img = cv2.imread(img_path)
	imgs = []
	imgs.append(img)
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img_otsu = blur_and_otsu(img_gray, blur_val=9)
	img_denoised, img_draw, contours = denoise(img_otsu)
	imgs.append(img_draw)
	imgs.append(img_denoised)
	show_imgs(imgs, cmap=None)
	return img_otsu



def preprocess_img(img):
	# 灰度模式
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# 中值滤波
	img_gray = cv2.medianBlur(img_gray, 3) #模板大小3*3

	# 二值化
	if img_gray.mean() > 100:
		ret, img_bin = cv2.threshold(img_gray, 0, 1, cv2.THRESH_OTSU)
	else:
		img_bin = (img_gray > 50).astype(np.uint8)  # 捕获要消去的背景，让文字呈现黑色
		for i in range(10):
			img_bin = cv2.medianBlur(img_bin, 3)

	img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_ERODE, np.ones((7, 7)))

	return img_bin

def run_img(img_path, resized_img_width=320, resized_img_height=160, show_plt=False, show_cv=False):
	logging.debug("Processing image: {}".format(img_path))
	img = cv2.imread(img_path)
	img = cv2.resize(img, (resized_img_width, resized_img_height))
	img_bin = preprocess_img(img)
	img_width  = img.shape[1]
	img_height = img.shape[0]

	# 轮廓查找
	contours, hierarchy = cv2.findContours(img_bin, 2, 2)
	sorted_contours = sorted(map(cv2.boundingRect, contours))

	# 图像切割
	i = 0
	last_rect_width = None
	# x, y 就是直角坐标系的数值
	for x, y, w, h in sorted_contours:
		# 筛选切割图
		if x>0 and w*h>img_width*img_height/(10*10):
			if not last_rect_width or not x+w < last_rect_width[-1]:
				i += 1
				# 原图矩形标注
				cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
				# 保存切割图
				cv2.imwrite("result/char_{}.png".format(i), img_bin[y: y+h, x: x+w] * 255)
				last_rect_width = (x, x+w)

	if show_plt == True:
		plt.imshow(img)
		plt.show()

	if show_cv == True:
		cv2.imshow("data.png", img)
		cv2.waitKey()

	return np.hstack([img, cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB) * 255])

def run_dir(img_dir_from, img_dir_to):
	if not os.path.exists(img_dir_to):
		os.mkdir(img_dir_to)
	IMGS_CNT = 5

	imgs = []
	for img_seq, img_name in enumerate(os.listdir(img_dir_from)):
		img_path = os.path.join(img_dir_from, img_name)
		img = run_img(img_path, IMG_WIDTH, IMG_HEIGHT)
		imgs.append(img)
		if img_seq % IMGS_CNT == IMGS_CNT - 1:
			cv2.imwrite("{}/{}.png".format(img_dir_to, time.time()), np.vstack(imgs))
			imgs = []
			# show_imgs(img_stack, img_width=IMG_WIDTH, img_height=IMG_HEIGHT * IMGS_CNT)
	else:
		if imgs:
			cv2.imwrite("{}/{}.png".format(img_dir_to, int(time.time())), np.vstack(imgs))


def run_by_otsu_step_test(img_path):
	img = cv2.imread(img_path, 0)
	imgs = [img]
	for blur_val in range(1, 20, 2):
		imgs.extend([blur_and_otsu(img, blur_val)])
	show_imgs(np.vstack(imgs), cmap="gray")

# run_by_otsu_step_test(img_path)
