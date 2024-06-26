# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:48:52 2024

@author: renxi
"""

import torch
import torch.nn as nn


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.Conv1x1(U)         
        q = self.norm(q)
        return U * q  


if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    s_se = sSE(c)
    print("in shape:", in_tensor.shape)
    out_tensor = s_se(in_tensor)
    print("out shape:", out_tensor.shape)
