# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 9:10
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

from settings import *
import pandas as pd
import shutil


DIR_NAME = "captcha_img_data"
FROM_DIR = os.path.join(RESOURCE_DIR, DIR_NAME)
TO_DIR   = os.path.join(RESOURCE_DIR, DIR_NAME + "_renamed_v2")

type_dict = dict()
for type_dir_seq, type_dir_name in enumerate(os.listdir(FROM_DIR)):
	print("Started migrating Dir {}".format(type_dir_name))
	type_to_dir_path = os.path.join(TO_DIR, type_dir_name)
	os.makedirs(type_to_dir_path, exist_ok=True)

	type_dir_path = os.path.join(FROM_DIR, type_dir_name)
	img_map_from_seq    = dict((i, img_name) for i, img_name in enumerate(os.listdir(type_dir_path)))
	img_map_from_name   = dict((j, i) for i, j in img_map_from_seq.items())
	type_dict[type_dir_name] = img_map_from_seq

	for img_name in os.listdir(type_dir_path):
		img_path = os.path.join(type_dir_path, img_name)
		shutil.copy(img_path, os.path.join(type_to_dir_path, "{}_{}.jpg".format(type_dir_seq, img_map_from_name[img_name])))


df = pd.DataFrame(type_dict)
df_path = os.path.join(TO_DIR, "mapping.csv")
df.to_csv(df_path, encoding="utf_8_sig", index=False)
print("Finished outputting the mapping file to {}".format(df_path))