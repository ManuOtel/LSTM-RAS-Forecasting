"""RAS Digitalization project.

This module contains the script that is doing the main training for the forecasting AI model.

@Author: Emanuel-Ionut Otel
@Company: Billund Aquaculture A/S
@Created: 2022-02-05
@Contact: manuotel@gmail.com
"""

#### ---- IMPORTS AREA ---- ####
import os, gc, tqdm, pickle, torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from data_handler import read_data
from sequence_data import SequenceDataset
from model import LSTMModel
#### ---- IMPORTS AREA ---- ####


#### ---- GLOBAL INIT AREA ---- ####    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
gc.collect()
BATCH_SIZE = 10
SEQUENCE_LENGTH = 10
NUM_FEATURES = 9
NUM_OUTPUT = 1
cwd = os.getcwd()
unique_names = ['P1', 'P2', 'S1', 'S2', 'Y1', 'Y2']
#### ---- GLOBAL INIT AREA ---- ####


if __name__ == '__main__':
    os.chdir('..')
    for name in unique_names:
        data = read_data(os.getcwd()+'/named_data/' + name + '_data.xlsx')

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

        model = LSTMModel(input_dim=NUM_FEATURES, hidden_dim=64, layer_dim=2, output_dim=1)
        model = model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, (30,))

        train_losses = []
        train_accuracies = []
        lrs = []

        loader_train = DataLoader(sequence_loader_train, shuffle=False, batch_size=BATCH_SIZE, num_workers=4, drop_last=True)
        loader_test = DataLoader(sequence_loader_test, batch_size=1, shuffle=False)

        # Initialize hidden state with zeros
        hn = torch.zeros(model.layer_dim, BATCH_SIZE, model.hidden_dim)
        # Initialize cell state
        cn = torch.zeros(model.layer_dim, BATCH_SIZE, model.hidden_dim)

        model.train()

        pbar = tqdm.tqdm(range(500))
        for epoch in pbar:
            # Initialize hidden state with zeros
            hn = torch.zeros(model.layer_dim, BATCH_SIZE, model.hidden_dim)
            # Initialize cell state
            cn = torch.zeros(model.layer_dim, BATCH_SIZE, model.hidden_dim)
            losses = []
            correct = total = 0
            for x, y in loader_train:
                x, y = x.to(device), y.to(device)
                y_pred, hn, cn = model(x, hn, cn)
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
            lrs.append(next(iter(opt.param_groups))['lr'])
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            pbar.set_description(f'loss: {train_loss:.3f}')

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

        # Initialize hidden state with zeros
        hn = torch.zeros(model.layer_dim, 1, model.hidden_dim)
        # Initialize cell state
        cn = torch.zeros(model.layer_dim, 1, model.hidden_dim)

        for x, y in loader_test:
            model.eval()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred, hn, cn = model(x, hn, cn)
            y = y.flatten()
            y_pred = y_pred.flatten()
            y_hat.append(y_pred.cpu().detach().numpy())
            y_target.append(y.cpu().detach().numpy())
        food_scaler_name = os.getcwd()+'/data/scaler_feed.pkl'
        scaler = pickle.load(open(food_scaler_name, 'rb'))
        y_hat_n = scaler.inverse_transform(y_hat)
        y_target_n = scaler.inverse_transform(y_target)
        plt.plot(y_hat_n, label='Prediction')
        plt.plot(y_target_n, label='Real')
        plt.legend()
        plt.title(name)
        plt.show()
        os.system('clear')


#### ---- QUICK NOTES AREA ---- ####
# salinity, CO2(not good, very high standard deviation), oxigen, Temperature, pH
# bacteria makes process ammonium->nitrit->nitrate and produce acid that reduce pH level
# liter/kg --- fish kW/kg fish
# oxigen shows the quality of food
# pH level change can show the performance of the biological system
# CO2 -> TAN, no need for NITRIT and NITRATE
#### ---- QUICK NOTES AREA ---- ####