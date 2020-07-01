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

class LSTMAM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, bidirectional):
        super(LSTMAM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        if bidirectional:
            self.output_layer = nn.Linear(hidden_size*2, output_size)
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)

        self.lstm = nn.LSTM(input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers ,
            batch_first = True,
            dropout = self.dropout,
            bidirectional = self.bidirectional)

    def forward(self, data):
        self.lstm.flatten_parameters()
        output, (h,c) = self.lstm(data)
        output = self_output_layer(output)      
  
        return output

