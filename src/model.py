from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building the LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        nn
        self.relu = nn.LeakyReLU()

    def forward(self, x, hn, cn):
        out, (hn, cn) = self.lstm(x, (hn.detach(), cn.detach()))

        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out, hn, cn
