import sys
sys.path.append('classification/')

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, font_output_data, font_target_data, use_hard_mining_bool=False):
        batch_size_data = font_output_data.size(0)
        font_output_data = self.sigmoid(font_output_data)
        font_output_data = font_output_data.view(-1)
        font_target_data = font_target_data.view(-1)
        pos_loss, neg_loss, font_loss = self.forward_func(font_output_data, font_target_data, use_hard_mining_bool, batch_size_data)
        return [font_loss, pos_loss, neg_loss]

    def forward_func(self, font_output_data, font_target_data, use_hard_mining_bool, batch_size_data):
        pos_index_num = font_target_data == 1
        neg_index_num = font_target_data == 0
        pos_target_data = font_target_data[pos_index_num]
        pos_output_data = font_output_data[pos_index_num]
        
        if use_hard_mining_bool:
            num_hard_pos = max(2, int(0.2 * batch_size_data))
            if len(pos_output_data) > num_hard_pos:
                pos_output_data, pos_target_data = hard_mining_func(pos_output_data, pos_target_data, num_hard_pos, largest_bool=False, start_num=int(num_hard_pos/4))
        if len(pos_output_data):
            pos_loss = self.classify_loss(pos_output_data, pos_target_data) * 0.5
        else:
            pos_loss = 0
        neg_output = font_output_data[neg_index_num]
        neg_target = font_target_data[neg_index_num]
        if use_hard_mining_bool:
            num_hard_neg = max(num_hard_pos, 2)
            if len(neg_output) > num_hard_neg:
                neg_output, neg_target = hard_mining_func(neg_output, neg_target, num_hard_neg, largest_bool=True, start_num=int(num_hard_pos/4))
        if len(neg_output):
            neg_loss = self.classify_loss(neg_output, neg_target) * 0.5
        else:
            neg_loss = 0

        font_loss = pos_loss + neg_loss
        return pos_loss,neg_loss,font_loss
        


def hard_mining_func(neg_output_data, neg_labels_data, num_hard_data, largest_bool=True, start_num=0):
    
    _, idcs = torch.topk(neg_output_data, min(num_hard_data, len(neg_output_data)), largest=largest_bool)
    start_num = 0
    idcs = idcs[start_num:]
    neg_output_data = torch.index_select(neg_output_data, 0, idcs)
    neg_labels_data = torch.index_select(neg_labels_data, 0, idcs)
    return neg_output_data, neg_labels_data
