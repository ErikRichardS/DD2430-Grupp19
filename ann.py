import torch
import torch.nn as nn



# The residual block's output is fed both into the next layer
# concatenated onto the input of a later layer
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU()
		)

	def forward(self, x):
		out = self.encoder(x)

		return out


# The downsampling block cosnists of 5 residual blocks.
# The output of each residual block are concatenated together
# and creates the second output of the block.
class DownsamplingBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.res_block1 = ResidualBlock(in_channels, out_channels)
		self.res_block2 = ResidualBlock(out_channels, out_channels)
		self.res_block3 = ResidualBlock(out_channels, out_channels)
		self.res_block4 = ResidualBlock(out_channels, out_channels)
		self.res_block5 = ResidualBlock(out_channels, out_channels)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


	def forward(self, x):
		out = self.res_block1(x)
		res = out

		out = self.res_block2(out)
		res = torch.cat( (res, out), dim=1 )

		out = self.res_block3(out)
		res = torch.cat( (res, out), dim=1 )

		out = self.res_block4(out)
		res = torch.cat( (res, out), dim=1 )

		out = self.res_block5(out)
		res = torch.cat( (res, out), dim=1 )

		out = self.pool(out)

		return out, res


# The upsampling block takes two inputs, the output of 
# the directly previous layer and the residual output.
class UpsamplingBlock(nn.Module):
	def __init__(self, in_channels, res_channels, out_channels):
		super().__init__()
		self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.decoder = nn.Sequential(
			nn.Conv2d(out_channels + (res_channels*5), out_channels, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU()
		)

	def forward(self, x, res):
		out = self.upsample(x)

		out = torch.cat( (out, res), dim=1 )

		out = self.decoder(out)

		return out



class U_Net(nn.Module):
	def __init__(self):
		super(U_Net, self).__init__()
		self.encoder1 = DownsamplingBlock(1, 32)
		self.encoder2 = DownsamplingBlock(32, 64)
		self.encoder3 = DownsamplingBlock(64, 128)
		self.encoder4 = DownsamplingBlock(128, 256)
		self.encoder5 = DownsamplingBlock(256, 512)

		self.bottleneck= nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=1),
			nn.ReLU(),
			nn.Conv2d(1024, 1024, kernel_size=1),
			nn.ReLU(),
		)
		
		self.decoder1 = UpsamplingBlock(1024, 512, 512)
		self.decoder2 = UpsamplingBlock(512, 256, 256)
		self.decoder3 = UpsamplingBlock(256, 128, 128)
		self.decoder4 = UpsamplingBlock(128, 64, 64)
		self.decoder5 = UpsamplingBlock(64, 32, 32)

		self.output_layer = nn.Sequential(
			nn.Conv2d(32, 1, kernel_size=1),
			nn.Sigmoid()
		)

		self.cuda()

	def forward(self, x):
		enc1, res1 = self.encoder1(x)
		enc2, res2 = self.encoder2(enc1)
		enc3, res3 = self.encoder3(enc2)
		enc4, res4 = self.encoder4(enc3)
		enc5, res5 = self.encoder5(enc4)

		bot = self.bottleneck(enc5)

		dec1 = self.decoder1(bot, res5)
		dec2 = self.decoder2(dec1, res4)
		dec3 = self.decoder3(dec2, res3)
		dec4 = self.decoder4(dec3, res2)
		dec5 = self.decoder5(dec4, res1)

		out = self.output_layer(dec5)

		return out
