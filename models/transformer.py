"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import sys
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, dim_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, dim_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayerWithConv1d(nn.Module): 

    """
      Input and output shape: seqlen x batch_size x dim
    """
    def __init__(self, dim_model, nheads, dim_feedforward, dropout, kernel_size, stride):
        super(TransformerEncoderLayerWithConv1d, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(dim_model, nheads, dim_feedforward, dropout)
        self.conv1d = nn.Conv1d(dim_model, dim_model, kernel_size, stride=stride, padding=1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
       output = self.encoder_layer(src, src_mask, src_key_padding_mask)
       output = F.relu(self.conv1d(output.permute(1, 2, 0)))

       return output.permute(2, 0, 1)

class TransformerAM(nn.Module):

    def __init__(self, dim_feat, 
                dim_model, 
                nheads, 
                dim_feedforward, 
                nlayers, 
                dropout, 
                output_size, 
                kernel_size=3,
                stride=1):
        super(TransformerAM, self).__init__()
        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        self.input_layer = nn.Linear(dim_feat, dim_model)
        self.output_layer = nn.Linear(dim_model, output_size)
        encoder_norm = nn.LayerNorm(dim_model)
        encoder_layer = TransformerEncoderLayerWithConv1d(dim_model, nheads, dim_feedforward, dropout, kernel_size, stride)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers, norm=encoder_norm)

    def forward(self, data, src_mask=None, src_key_padding_mask=None):
        input = self.input_layer(data)
        #input = self.pos_encoder(input)
        output = self.transformer(input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.output_layer(output)

        return output


