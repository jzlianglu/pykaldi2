
import os
import sys
import numpy as np
import torch as th
import torch.nn as nn

class LSTMStack(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(LSTMStack, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers ,
            batch_first = True,
            dropout = self.dropout,
            bidirectional = self.bidirectional)

    def forward(self, data):
        self.lstm.flatten_parameters()
        output, (h,c) = self.lstm(data)
        
        return output, (h,c)


class NnetAM(nn.Module):
    
    def __init__(self, nnet, hidden_size, output_size):
        super(NnetAM, self).__init__()

        self.nnet = nnet
        self.output_size = output_size
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        nnet_output, (h,c) = self.nnet(data)
        output = self.output_layer(nnet_output)

        return output


