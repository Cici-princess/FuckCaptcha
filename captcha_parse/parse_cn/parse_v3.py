# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 19:38
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------
from general import *



img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data_renamed_v2\Type_6\5_205.jpg'

img = cv2.imread(img_path)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

H, W = 40, 100
target = img[0:H, 0: W]
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram and apply backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)

# threshold and binary AND
ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)

res = np.vstack((cv2.resize(img, (W, H)), target, thresh, res))

show_img(res)
