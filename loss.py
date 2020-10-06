import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




def invert(t):

    return (t-1)*-1



class FocalLoss(nn.Module):

    def __init__(self, weight_negative=0.75, weight_positive=50, gamma=2):
        super(FocalLoss, self).__init__()
        self.w_neg = weight_negative
        self.w_pos = weight_positive
        self.gamma = gamma

    def forward(self, input, target):


       
        p = input*target + invert(input)*invert(target)

        
        pos = -self.w_pos * torch.pow( (1 - p), self.gamma ) * torch.log(p)
        neg = -self.w_neg * torch.pow(    p,    self.gamma ) * torch.log(1 - p)
        loss = pos + neg

        return torch.mean(loss)