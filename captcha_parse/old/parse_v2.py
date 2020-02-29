# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 8:09
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------


def verify_one(img):
	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)     # ！灰度
	if img.shape != IMG_SPLIT_SIZE:
		img = cv2.resize(img, IMG_SPLIT_SIZE)                 # ！转换成目标大小
	img = trans(img).to('cuda')
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
	img_info = {"value": value, "probability": probability}

	if probability < 0.5:
		logging.warning(img_info)
	else:
		logging.info(img_info)
	return value