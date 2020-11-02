import torch 
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from time import time
import numpy as np


from ann import *
from loss import *
from data_manager import *
from random import uniform
import copy



# Hyper Parameters
num_epochs = 5
batch_size = 4

max_lr = 1e-4
min_lr = 1e-6
learning_rate = []
for i in range(10):
	learning_rate.append(uniform(min_lr, max_lr))



net_untrained = U_Net()

trn_dataset = get_training_data() # Training data


# Loaders handle shufflings and splitting data into batches
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)



for lr in learning_rate:
	print("testing learning rate: %0.5f" % lr)
	
	net = copy.deepcopy(net_untrained)

	# Criterion calculates the error/loss of the output
	# Optimizer does the backprop to adjust the weights of the NN
	criterion = FocalLoss() # nn.BCELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr) 

	best_loss = 1e10

	for epoch in range(num_epochs):
		t1 = time()

		loss_sum = 0

		for i, (data, labels) in enumerate(trn_loader):
			# Load data into GPU using cuda
			data = data.cuda()
			labels = labels.cuda()

			# Forward + Backward + Optimize
			optimizer.zero_grad()
			#out = net(data)
			#outputs = torch.sigmoid(out)

			outputs = net(data)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			if torch.isnan(loss):
				print(out[0])

			loss_sum += loss

		t2 = time()
		if loss_sum < best_loss:
			best_loss = loss_sum
		print("Epoch time : %0.3f m \t Loss : %0.3f" % ( (t2-t1)/60 , loss_sum ))

	print("Learning rate: %0.3f \t Best loss: %0.3f" % (lr, best_loss))