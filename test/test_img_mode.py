# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/1 4:09
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

from general import *
from PIL import Image

img_path = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data_renamed_v2_output\Type_1\0_0_0.jpg'

img_cv = cv2.imread(img_path)
img_pil = Image.open(img_path)