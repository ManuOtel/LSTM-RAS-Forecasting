import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn import functional as F
import numpy as np
from data_handler import read_data
import os
import gc
from sequence_data import SequenceDataset
from another_one import ShallowRegressionLSTM, LSTMModel
import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()
gc.collect()

BATCH_SIZE = 128
SEQUENCE_LENGTH = 130
NUM_FEATURES = 9
NUM_OUTPUT = 1

cwd = os.getcwd()

unique_names = ['P1', 'P2', 'S1', 'S2', 'Y1', 'Y2']

for name in unique_names:
    data = read_data('../named_data/' + name + '_data.xlsx')

    y = data['feed']
    x = data.drop(['feed', 'Unnamed: 0', 'name', 'date', 'pH'], axis=1)

    x_train = torch.Tensor(np.array(x)).to(device, non_blocking=True)
    x_test = x_train.to(device, non_blocking=True)

    y_train = torch.Tensor(np.array(y)).to(device, non_blocking=True)
    y_test = y_train.to(device, non_blocking=True)

    data_train = TensorDataset(x_train, y_train)
    data_test = TensorDataset(x_test, y_test)

    sequence_loader_train = SequenceDataset(x_train, y_train, sequence_length=SEQUENCE_LENGTH)
    sequence_loader_test = SequenceDataset(x_test, y_test, sequence_length=SEQUENCE_LENGTH)

    model = LSTMModel(input_dim=NUM_FEATURES, hidden_dim=128, layer_dim=1, output_dim=1)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, (30,))

    train_losses = []
    train_accuracies = []
    lrs = []

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0)
    loader_train = DataLoader(sequence_loader_train, shuffle=False, **loader_kwargs)
    loader_test = DataLoader(sequence_loader_test, batch_size=1, shuffle=False)

    model.train()

    pbar = tqdm.tqdm(range(1500))
    for epoch in pbar:
        losses = []
        correct = total = 0
        for x, y in loader_train:
            x, y = x.to(device), y.to(device)
            # print(x.shape)
            # print(y.shape)
            y_pred = model(x)
            y_pred = y_pred.flatten()
            loss = F.mse_loss(y_pred, y)

            loss.backward()
            opt.step()
            opt.zero_grad()

            losses.append(loss.item())
            correct += ((torch.sigmoid(y_pred) > 0.5) == y).sum().item()
            total += len(x)
        train_loss = np.mean(losses)
        train_acc = correct / total
        sched.step()
        # if train_loss <= 0.001:
        # break

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

    torch.save(model.state_dict(), os.getcwd() + '/models/' + name + '_model_no_break2.pt')

    for x, y in loader_test:
        model.eval()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_pred = model(x)
        y = y.flatten()
        y_pred = y_pred.flatten()
        y_hat.append(y_pred.cpu().detach().numpy())
        y_target.append(y.cpu().detach().numpy())

    plt.plot(y_hat, label='Prediction')
    plt.plot(y_target, label='Real')
    plt.legend()
    plt.title(name)
    plt.show()
    os.system('clear')

# salinity, CO2(not good, very high standard deviation), oxigen, Temperature, pH
# bacteria makes process ammonium->nitrit->nitrate and produce acid that reduce pH level
# liter/kg --- fish kW/kg fish
# oxigen shows the quality of food
# pH level change can show the performance of the biological system
