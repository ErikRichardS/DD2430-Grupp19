import torch
import torch.nn as nn



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


class U_Net(nn.Module):
	def __init__(self):
		super(U_Net, self).__init__()
		self.encoder1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
		)

		self.encoder2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
		)

		self.encoder3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(128),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
		)

		self.encoder4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(256),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
		)

		self.encoder5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
		)

		self.bottleneck= nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 1024, kernel_size=1),
			nn.ReLU(),
			nn.Conv2d(1024, 1024, kernel_size=1),
			nn.ReLU(),
			nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
		)

		
		self.decoder1 = nn.Sequential(
			nn.Conv2d(1024, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
		)

		self.decoder2 = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(256),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
		)

		self.decoder3 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(128),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
		)

		self.decoder4 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
		)

		self.decoder5 = nn.Sequential(
			nn.Conv2d(64, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 3, kernel_size=3, padding=1),
			nn.Sigmoid()
		)



		self.cuda()

	def forward(self, x):
		enc1 = self.encoder1(x)
		enc2 = self.encoder2(enc1)
		enc3 = self.encoder3(enc2)
		enc4 = self.encoder4(enc3)
		enc5 = self.encoder5(enc4)

		bot = self.bottleneck(enc5)

		cat0 = torch.cat( (bot, enc5), dim=1)
		dec1 = self.decoder1(cat0)

		cat1 = torch.cat( (dec1, enc4), dim=1)
		dec2 = self.decoder2(cat1)

		cat2 = torch.cat( (dec2, enc3), dim=1)
		dec3 = self.decoder3(cat2)

		cat3 = torch.cat( (dec3, enc2), dim=1)
		dec4 = self.decoder4(cat3)

		cat4 = torch.cat( (dec4, enc1), dim=1)
		dec5 = self.decoder5(cat4)

		return dec5
