import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn import functional as F
import numpy as np
from data_handler import read_data
import os
from sequence_data import SequenceDataset
from another_one import ShallowRegressionLSTM, LSTMModel
import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

cwd = os.getcwd() + '/data'

data = read_data('../named_data/P1_data.xlsx')

y = data['feed']
x = data.drop(['feed', 'Unnamed: 0', 'name', 'date', 'pH'], axis=1)


BATCH_SIZE = 64
SEQUENCE_LENGTH = 4
NUM_FEATURES = 9
NUM_OUTPUT = 1

# print(x)

x_train = torch.Tensor(np.array(x)).to(device, non_blocking=True)  # .unsqueeze_(-1).expand(176, 9, 4)
x_test = x_train.to(device, non_blocking=True)

# print(x_train.shape)
# print(x_train)

y_train = torch.Tensor(np.array(y)).to(device, non_blocking=True)
y_test = y_train.to(device, non_blocking=True)

data_train = TensorDataset(x_train, y_train)
data_test = TensorDataset(x_test, y_test)

sequence_train = SequenceDataset(x_train, y_train, sequence_length=4)
sequence_test = SequenceDataset(x_test, y_test, sequence_length=4)

# model = LSTM().to(device)
# model = LSTM1(num_classes=1, input_size=9, hidden_size=9, num_layers=128, seq_length=64).to(device, non_blocking=True)
# model = LSTM().to(device)
# model = LSTMModel(input_dim=9, hidden_dim=64, layer_dim=2, output_dim=1).to(device)

model = LSTMModel(input_dim=NUM_FEATURES, hidden_dim=1024, layer_dim=1, output_dim=4)
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-1)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, (30,))

train_losses = []
train_accuracies = []
lrs = []

loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0)
loader_train = DataLoader(data_train, shuffle=False, **loader_kwargs)
loader_test = DataLoader(data_test, batch_size=1, shuffle=False)

model.train()

pbar = tqdm.tqdm(range(1500))
for epoch in pbar:
    losses = []
    correct = total = 0
    for x, y in loader_train:
        if x.shape[0] < BATCH_SIZE:
            pad = torch.zeros(int(BATCH_SIZE - x.shape[0]), NUM_FEATURES).to(device)
            x = torch.cat([x, pad], dim=0).to(device)
        if y.shape[0] < BATCH_SIZE:
            pad = torch.zeros(int(BATCH_SIZE - y.shape[0])).to(device)
            y = torch.cat([y, pad], dim=0).to(device)

        x = x.view(int(BATCH_SIZE/SEQUENCE_LENGTH), SEQUENCE_LENGTH, NUM_FEATURES)
        y = y.view(int(BATCH_SIZE/SEQUENCE_LENGTH), SEQUENCE_LENGTH, NUM_OUTPUT)

        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        # print(y_pred.shape, ' DICK ', y.shape)
        y = y.view(16, 4)
        loss = F.mse_loss(y_pred.flatten(), y.flatten())

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

torch.save(model.state_dict(), os.getcwd()+'/models/el_model_very_good.pt')

for x, y in loader_test:
    model.eval()

    x = x.view(1, 1, NUM_FEATURES)
    y = y.view(1, 1, NUM_OUTPUT)

    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    y_pred = model(x)
    y = y.flatten()
    y_pred = y_pred.flatten()
    y_hat.append(y_pred.cpu().detach().numpy())
    y_target.append(y.cpu().detach().numpy())
    # print(y.shape, ' DICK ', y_pred.shape)

plt.plot(y_hat, label='Prediction')
plt.plot(y_target, label='Real')
plt.legend()
plt.show()
