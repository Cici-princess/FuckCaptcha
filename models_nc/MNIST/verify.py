# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 4:16
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------
from pytorch.model.train import MarkNet
"""
[pytorch用LeNet5识别Mnist手写体数据集(训练+预测单张输入图片代码)_人工智能_ZJE-CSDN博客](https://blog.csdn.net/u014453898/article/details/90707987 )
"""

from general import *

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


IMG_SIZE = (28, 28)
MODEL_PATH = os.path.join(DATA_DIR, "MNIST", 'model.pt')

IMG_DIR = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data_output\TrainingType_1_1582924353.0918396'
IMG_PATH = r'F:\MyProjects\PycharmProjects\FuckCaptcha\data\captcha_parse_v2\TrainingType_1_1582908368.9129198\0_1582907687_0.png'


trans = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])


def verify_one(img_path):
	if isinstance(img_path, str):
		img = cv2.imread(img_path)
	else:
		img = img_path
	img = cv2.resize(img, IMG_SIZE)                 # ！转换成目标大小
	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)     # ！灰度
	img = trans(img).to(device)
	"""
	扩展维度：
	模型中数据是4维的：  [batch_size,通道,长，宽]
	普通图片只有三维：   [通道,长，宽]
	扩展后为：           [1，1，28，28]
	"""
	img = img.unsqueeze(0)                          # 图片扩展多一维,
	output = model(img)
	prob = F.softmax(output, dim=1)
	prob = Variable(prob)
	prob_array = prob.cpu().numpy()   # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式

	probability = prob_array.max()
	value = np.argmax(prob_array)      # 选出概率最大的一个
	img_info = {"path": img_path, "value": value, "probability": probability}

	if probability < 0.5:
		logging.warning(img_info)
	else:
		logging.info(img_info)
	return value


def verify_dir(img_dir):
	from collections import defaultdict
	result_dict = defaultdict(dict)
	for img_name in os.listdir(img_dir):
		img_path = os.path.join(img_dir, img_name)
		img_value = verify_one(img_path=img_path)

		# TODO: 这里的文件名需要重新设置
		img_raw_name, img_seq = img_name.split("_")
		result_dict[img_raw_name][img_seq] = img_value

	for img_raw_name in list(result_dict):
		img_concat_value = "".join(map(lambda x: str(x[1]), sorted(result_dict[img_raw_name].items(), key=lambda x: x[0])))
		result_dict[img_raw_name] = img_concat_value

	import json, time
	json.dump(result_dict, open("result_1_{}.json".format(time.time()), "w", encoding="utf-8"), ensure_ascii=False, indent=4)

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = torch.load(MODEL_PATH).to(device)
	model.eval()                                    # 把模型转为test模式

	# verify_one(r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data_renamed_v2_output\Type_1\0_0_2.jpg')
	verify_dir(r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data_renamed_v2_output\Type_1')

