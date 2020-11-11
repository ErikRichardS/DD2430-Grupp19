import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.transforms.functional as F

from time import time
import numpy as np


from ann import *
from loss import *
from data_manager import *
from eval import compute_metrics, formalize_skeleton


# Hyper Parameters
num_epochs = 100
batch_size = 4
learning_rate = 1e-4
#weight_decay = 1e-7
learning_decay = 0.9


net = U_Net()  # torch.load("skeleton_net.pt")
trn_dataset, vld_dataset = get_training_data()  # Training and validation data


# Loaders handle shufflings and splitting data into batches
trn_loader = torch.utils.data.DataLoader(
    trn_dataset, batch_size=batch_size, shuffle=True)
vld_loader = torch.utils.data.DataLoader(vld_dataset, batch_size=1)


# Criterion calculates the error/loss of the output
# Optimizer does the backprop to adjust the weights of the NN
criterion = FocalLoss()  # nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


f1_best = 0
for epoch in range(num_epochs):
    t1 = time()  # Get starting time of epoch

    # Train the network on the training data
    loss_sum = 0
    for i, (data, labels) in enumerate(trn_loader):
        # Load data into GPU using cuda
        data = data.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(data)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Update the wights of the network
        loss.backward()
        optimizer.step()

        # Add loss to total loss of epoch
        loss_sum += loss

    # Test the network on the validation data
    f1_total = 0
    total = 0
    for i, (data, labels) in enumerate(vld_loader):
        data = data.cuda()

        # Detach output tensor so memory is not filled with possible gradient calculations
        outputs = net(data).detach()

        # Calculate precision, recall, and f1-score
        pre, rec, f1 = compute_metrics(labels, outputs.cpu())

        f1_total += f1
        total += 1

    f1_mean = f1_total / total

    t2 = time()  # Get ending time of epoch
    print("Epoch time : %0.3f m \t Loss : %0.3f \t F1 mean : %0.3f" %
          ((t2-t1)/60, loss_sum, f1_mean))

    # If new f1 score is the best so far, save network
    if f1_best < f1_mean:
        torch.save(net, "skeleton_net.pt")
        f1_best = f1_mean
