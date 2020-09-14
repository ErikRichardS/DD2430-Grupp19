import torch
import torch.nn as nn


# A CNN based on the VGG16 architecture
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			#nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			#nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 3, kernel_size=3, padding=1),
			#nn.MaxPool2d(kernel_size=2, stride=2)
		)
		
		self.cuda()

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)

		return out

