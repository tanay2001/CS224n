#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h


class Highway(nn.Module):
    def __init__(self,embedding_dim):
        super(Highway, self).__init__()
        self.xproj = nn.Linear(embedding_dim, embedding_dim ,bias=True)
        self.xgate = nn.Linear(embedding_dim, embedding_dim, bias=True)


    def forward(self, source : torch.Tensor) -> torch.Tensor:
        xproj = F.relu(self.xproj(source))
        xgate = torch.sigmoid(self.xgate(source))
                
        return torch.mul(xproj,xgate) + torch.mul(source, 1-xgate)




### END YOUR CODE 

