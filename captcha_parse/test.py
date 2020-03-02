# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 23:19
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------


from captcha_parse.parse_v3 import parse_img
from pytorch.model.train import MarkNet


img_path = 'samples/0_0889938.jpg'

raw_img, split_imgs, value = parse_img(
	img_path, img_dir_to=".", show=True, verify=True)

print("Value: {}".format(value))