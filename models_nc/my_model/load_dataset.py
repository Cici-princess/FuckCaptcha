# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 21:11
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
	]
)


TRAIN_DIR = r'F:\MyProjects\PycharmProjects\FuckCaptcha\src\captcha_img_data_renamed_v2_output'

EPOCH = 3
TRAIN_BATCH_SIZE = 64
NUM_WORKER = 1

from settings import *
from pytorch.model.train import MarkNet


if __name__ == '__main__':
    char_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=TRANSFORM)
    char_train_loader = DataLoader(char_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)

    model = MarkNet()   # type: MarkNet
    model = model.to(DEVICE)

    for epoch in range(EPOCH):
        for i, (data, target) in enumerate(char_train_loader, start=1):
            data = Variable(data).to(DEVICE)
            target = Variable(target).to(DEVICE)
            """
            这个data里，其实是batch_size个tensor的列表
            如果想展示原图像，可以：
            data -> 从CUDA中移到CPU -> 转换成np数组 -> 转换轨道 -> 获取其中的某个图像 
            img = data.cpu().numpy().transpose()[0]
            show_img(img)
            """

            model.run_training(char_train_loader)


