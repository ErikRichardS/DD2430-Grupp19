import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




def invert(t):
    return (t*-1)+1



class FocalLoss(nn.Module):

    def __init__(self, weight_negative=0.75, weight_positive=50, gamma=2):
        super(FocalLoss, self).__init__()
        self.w_neg = weight_negative
        self.w_pos = weight_positive
        self.gamma = gamma

    def forward(self, p, target):

        p_white = p*target + invert(target)

        p_black = p*invert(target)

        # Loss for white pixels
        positive = -self.w_pos * torch.pow( (1 - p_white), self.gamma ) * torch.log(p_white)
        #positive = torch.clamp(positive, 0, 1e10)

        # Loss for negative pixels
        negative = -self.w_neg * torch.pow(    p_black,    self.gamma ) * torch.log(1 - p_black)
        #negative = torch.clamp(negative, 0, 1e10)

        loss = positive + negative

        return torch.mean(loss)