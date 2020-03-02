# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/1 6:45
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

from settings import *

def pipe_img(img, funcs):
	def standard_img(img):
		if len(img.shape) <= 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		img = cv2.resize(img, (w, h))
		return img

	imgs = [img]
	h, w = img.shape[: 2]
	for func in funcs:
		img = func(img)
		imgs.append(img)
	show_img(np.vstack(list(map(standard_img, imgs))))
	return img

BLUE_THRESHOLD = 5
img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\sample_data\Type_1\0a62ec80-4e7b-11ea-a66f-001a7dda7113.jpg'
img = cv2.imread(img_path)

img_pipelines = [
	lambda x: cv2.GaussianBlur(x, (BLUE_THRESHOLD, BLUE_THRESHOLD), 0),
	lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
	lambda x: cv2.threshold(x, 127, 255, cv2.THRESH_OTSU)[1],
	lambda x: (255-x).astype(np.uint8) if x[0, 0] == 255 else x,
]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

MIN_LINE_LENGTH = 200
MAX_LENE_GAP = 3
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, MIN_LINE_LENGTH, MAX_LENE_GAP)

for line in lines:
	x1, y1, x2, y2 = line[0]
	cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.line(edges, (x1, y1), (x2, y2), 0, 2)
plt.imshow(np.vstack([img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)])); plt.show()