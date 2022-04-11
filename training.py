import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import numpy as np
from data_handler import read_data
import os
import pandas as pd
from neural_network import FeedPredictor
from neural_network4 import LSTM1
from neural_network2 import LSTM
from another_one import  LSTMModel
import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = os.getcwd() + '/data'

data = read_data('../named_data/P1_data.xlsx')

y = data['feed']
x = data.drop(['feed', 'Unnamed: 0', 'name', 'date', 'pH'], axis=1)


# print(x)

x_train = torch.Tensor(np.array(x)).to(device, non_blocking=True)
x_test = x_train.to(device, non_blocking=True)

print(x_train.shape)
print(x_train)

y_train = torch.Tensor(np.array(y)).to(device, non_blocking=True)
y_test = y_train.to(device, non_blocking=True)

data_train = SequenceDataset(x_train, y_train)
data_test = TensorDataset(x_test, y_test)

# model = LSTM().to(device)
# model = LSTM1(num_classes=1, input_size=9, hidden_size=9, num_layers=128, seq_length=64).to(device, non_blocking=True)
# model = LSTM().to(device)

model = LSTMModel(input_dim=9, hidden_dim=64, layer_dim=2, output_dim=1).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, (30,))

train_losses = []
train_accuracies = []
lrs = []

loader_kwargs = dict(batch_size=64, num_workers=0)
loader_train = DataLoader(data_train, shuffle=False, **loader_kwargs)
loader_test = DataLoader(data_test)

model.train()

pbar = tqdm.tqdm(range(40))
for epoch in pbar:
    state_h, state_c = model.init_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    losses = []
    correct = total = 0
    for x, y in loader_train:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        loss = F.mse_loss(y_pred, y)

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())
        correct += ((torch.sigmoid(y_pred) > 0.5) == y).sum().item()
        total += len(x)
    train_loss = np.mean(losses)
    train_acc = correct / total
    sched.step()

    # history
    lrs.append(next(iter(opt.param_groups))['lr'])
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    pbar.set_description(f'loss: {train_loss:.3f}')

# plot history
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].plot(train_losses, label='Train Loss')
axs[0].set_ylabel('loss')
axs[0].legend()
axs[1].plot(train_accuracies, label='Train Accuracy')
axs[1].set_ylabel('acc')
axs[1].legend()
axs[2].plot(lrs, label='Learning Rate')
plt.tight_layout()
plt.show()

y_hat = []
y_target = []

for x, y in loader_test:
    model.eval()
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    y_pred, (state_h, state_c) = model(x, (state_h, state_c))
    y_hat.append(torch.flatten(y_pred.cpu()).detach().numpy())
    y_target.append(torch.flatten(y.cpu()).detach().numpy())

plt.plot(y_hat, label='Prediction')
plt.plot(y_target, label='Real')
plt.legend()
plt.show()
