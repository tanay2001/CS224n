#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self,e_size , M_word,f, k =5):
        super(CNN, self).__init__()

        self.conv1D = nn.Conv1d(e_size,f, k ,bias=True)

        self.maxPool = nn.MaxPool1d(M_word-k+1)
    def forward(self, x_reshape: torch.Tensor) -> torch.Tensor:

        x_conv = self.conv1D(x_reshape)
        x_conv_out = self.maxPool(x_conv)

        return torch.squeeze(x_conv_out, -1)





### END YOUR CODE

