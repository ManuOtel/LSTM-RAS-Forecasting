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

        self.ffc1 = nn.Linear(15, 128)
        self.ffc2 = nn.Linear(128, 15)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 256)  # fully connected 1
        self.fc_2 = nn.Linear(256, 32)  # fully connected 1
        self.fc = nn.Linear(32, num_classes)  # fully connected last layer

        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        hidden = (torch.zeros(self.num_layers, self.hidden_size, device="cuda"), torch.zeros(self.num_layers, self.hidden_size, device="cuda"))
        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        # x = self.relu(self.ffc2(self.relu(self.ffc1(x))))
        output, _ = self.lstm(x, hidden)  # lstm with input, hidden, and internal state
        print(np.shape(output))
        output = output.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        print(np.shape(output))
        out = self.relu(output)
        out = self.fc_1(out)  # first Dense
        # out = self.relu(out)  # relu
        out = self.fc_2(out)  # first Dense
        # out = self.relu(out)
        out = self.relu(self.fc(out))
        return out
