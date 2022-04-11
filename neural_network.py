import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class FeedPredictor(nn.Module):
    def __init__(self, lstm_input=15, n_lstm_layers=10, dropout=0.):
        super().__init__()
        self.lstm = nn.LSTM(lstm_input, n_lstm_layers, 1, batch_first=True, dropout=dropout)
        self.lin = nn.Linear(n_lstm_layers, 64)
        self.lin2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.lstm(x, 64)
        x = self.lin(x)
        x = nn.ReLU(x)
        x = self.lin2(x)
        out = nn.ReLU(x)
        return out
