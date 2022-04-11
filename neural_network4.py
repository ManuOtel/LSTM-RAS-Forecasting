import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.1)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 64)  # fully connected 1
        self.fc_2 = nn.Linear(64, num_classes)  # fully connected 1
        self.relu = nn.ReLU()

    def forward(self, x, prev_state):
        # Propagate input through LSTM
        output, state = self.lstm(x, prev_state)
        output, state = self.lstm(output, state)  # lstm with input, hidden, and internal state
        output = output.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.fc_1(output)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # first Dense
        out = self.relu(out)
        return out, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, self.hidden_size),
                torch.zeros(self.num_layers, self.hidden_size))
