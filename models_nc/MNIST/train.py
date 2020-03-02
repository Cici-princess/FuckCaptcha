# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 4:07
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable


LEARNING_RATE       = 0.01  # 学习率
MOMENTUM            = 0.5   #
LOG_INTERVAL        = 10    # 日志记录周期
EPOCHS              = 2    # 迭代次数
TRAINING_BATCH_SIZE = 64    # 训练时每一批个数
TO_TEST_BATCH_SIZE  = 1000  # 测试时每一批个数

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 启用GPU

TRANSFORM = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
])


from settings import *

DATA_DIR   = DATA_DIR
MODEL_PATH = os.path.join(DATA_DIR, "MNIST", "model.pt")



class MarkNet(nn.Module):
	def __init__(self):
		super(MarkNet, self).__init__()
		self.conv1 = nn.Sequential(                 # input_size=(1*28*28)
			nn.Conv2d(1, 6, 5, 1, 2),               # padding=2保证输入输出尺寸相同
			nn.ReLU(),                              # input_size=(6*28*28)
			nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(6, 16, 5),
			nn.ReLU(),                              # input_size=(16*10*10)
			nn.MaxPool2d(2, 2)                      # output_size=(16*5*5)
		)
		self.fc1 = nn.Sequential(
			nn.Linear(16 * 5 * 5, 120),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(120, 84),
			nn.ReLU()
		)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		"""
		该功能是必须要重写的
		:param x:
		:return:
		"""
		x = self.conv1(x)
		x = self.conv2(x)
		# nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
		x = x.view(x.size()[0], -1)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x  # F.softmax(x, dim=1)

	def run_training(self, train_loader, optimizer=None, epoch=1):   # 定义每个epoch的训练细节
		self.train()    # 设置为trainning模式
		if not optimizer:
			optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)  # 初始化优化器

		for batch_idx, (data, target) in enumerate(train_loader):
			# 计算前要把变量变成Variable形式，因为这样子才有梯度
			data    = Variable(data.to(DEVICE))
			target  = Variable(target.to(DEVICE))

			optimizer.zero_grad()  # 优化器梯度初始化为零
			output = self(data)  # 把数据输入网络并得到输出，即进行前向传播
			loss = F.cross_entropy(output, target)  # 交叉熵损失函数
			loss.backward()  # 反向传播梯度
			optimizer.step()  # 结束一次前传+反传之后，更新参数

			if batch_idx % LOG_INTERVAL == 0:  # 准备打印相关信息
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					       100. * batch_idx / len(train_loader), loss.item()))

	def run_testing(self, test_loader, epoch=1):
		self.eval()         # 设置为test模式
		loss = 0            # 初始化测试损失值为0
		correct_cnt = 0     # 初始化预测正确的数据个数为0

		for data, target in test_loader:
			data    = Variable(data.to(DEVICE))
			target  = Variable(target.to(DEVICE))

			output = self(data)
			loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss 把所有loss值进行累加
			pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
			correct_cnt += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

		loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
		print('\nTest Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			epoch, loss, correct_cnt, len(test_loader.dataset),
			100. * correct_cnt / len(test_loader.dataset)))

	def save_model(self, model_path):
		torch.save(self, model_path)


if __name__ == '__main__':

	train_dataset = datasets.MNIST(DATA_DIR, train=True, transform=TRANSFORM, download=True)
	train_loader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True)

	test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=TRANSFORM)
	test_loader = DataLoader(test_dataset, batch_size=TO_TEST_BATCH_SIZE, shuffle=True)

	model = MarkNet()
	model = model.to(DEVICE)

	for epoch in range(1, EPOCHS + 1):  # 以epoch为单位进行循环
		model.run_training(train_loader, epoch=epoch)
		model.run_testing(test_loader, epoch=epoch)

	model.save_model(MODEL_PATH)


