from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pickle import dump

cwd = os.getcwd()
cwd = cwd + '/data'


def save_data(save_to_data, filename):
    save_to_data.to_excel(filename)


def one_hot_encode(normal_data, drop_date=False, ph_drop=False):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_normal_data = pd.DataFrame(encoder.fit_transform(normal_data[['name']]).toarray())
    encoded_data = normal_data.join(encoder_normal_data)
    encoded_data.drop('name', axis=1, inplace=True)
    if drop_date:
        encoded_data.drop('date', axis=1, inplace=True)
    if ph_drop:
        encoded_data.drop('pH', axis=1, inplace=True)
    return encoded_data


def normalize_data(full_data, save=True):
    aux_data = full_data.select_dtypes(include='number')
    normalized_data = pd.DataFrame()
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    for i in aux_data:
        x = aux_data[i].values.reshape(1, -1).astype(np.float)  # returns a numpy array
        # print(x)
        x_scaled = min_max_scaler.fit_transform(np.transpose(x)).astype(np.float)
        # print(x_scaled)
        normalized_data[i] = x_scaled.reshape(-1, ).astype(np.float)
        if save:
            dump(min_max_scaler, open('scaler_' + str(i) + '.pkl', 'wb'))
    full_data[normalized_data.columns] = normalized_data
    return full_data


def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def batch_finder(sector_data):
    batch_index = []
    last_value = 0  # sector_data["density"][1]
    value = True
    for idx in sector_data.index:
        i = sector_data["density"][idx]
        # print(i)
        der = abs(i - last_value)
        if der > 0:
            batch_index.append(idx)
    return batch_index


def plot_single_data(plot_data, figure_name, plot_batch):
    # fig = plt.figure()
    plt.subplots(constrained_layout=True)

    plt.subplot(811).set_title('Density', loc='right', fontsize=10)
    plt.plot(plot_data["density"])
    plt.title(figure_name)

    if plot_batch:
        batch_index = batch_finder(plot_data)
        for i in batch_index:
            plt.axvline(x=i, c='red')
    # print(batch_index)

    plt.subplot(812).set_title('Feed', loc='right', fontsize=10)
    plt.plot(plot_data["feed"])

    plt.subplot(813).set_title('Salt', loc='right', fontsize=10)
    plt.plot(plot_data["salt"])

    plt.subplot(814).set_title('Temp', loc='right', fontsize=10)
    plt.plot(plot_data["avg_temp"])

    plt.subplot(815).set_title('Ammonium', loc='right', fontsize=10)
    plt.plot(plot_data["ammonium"])

    plt.subplot(816).set_title('Nitrit', loc='right', fontsize=10)
    plt.plot(plot_data["nitrit"])

    plt.subplot(817).set_title('Nitrate', loc='right', fontsize=10)
    plt.plot(plot_data["nitrate"])

    plt.subplot(818).set_title('CO2', loc='right', fontsize=10)
    plt.plot(plot_data["co2"])

    # out = plt.get_figure()
    # print(out)

    plt.show()
    # return fig, ax


def plot_all_data(total_data, unique_name, plot_batch=False):
    figures = []
    for i in unique_name:
        # print(total_data[total_data["name"] == i])
        plot_single_data(total_data[total_data["name"] == i], i, plot_batch)
        # figures.append(fig)

    # app = qt.QApplication(sys.argv)
    # ui = MplMultiTab(figures=figures)
    # ui.show()
    # app.exec_()


def all_batches_finder(total_data, unique_name):
    all_batches = [None] * len(unique_name)
    for i in unique_name:
        # print(total_data[total_data["name"] == i])
        current_batch = batch_finder(total_data[total_data["name"] == i])
        all_batches[unique_name.index(i)] = current_batch
    return all_batches


def save_data_by_name(total_data, unique_name):
    for i in unique_name:
        named_data = total_data[total_data['name'] == i]
        save_data(named_data, 'named_data/' + i + '_data.xlsx')


def batch_data(total_data, unique_name, all_batches, missing_data=False):
    all_batch_data = pd.DataFrame()
    for i in unique_name:
        # print(all_batches[unique_name.index(i)])
        sector_data = total_data[total_data["name"] == i]
        # print(sector_data.loc[all_batches[unique_name.index(i)]])
        batch_sector_data = sector_data.loc[all_batches[unique_name.index(i)]]
        if missing_data:
            batch_sector_data['ammonium'] = batch_sector_data['ammonium'].interpolate()
            batch_sector_data['avg_temp'] = batch_sector_data['avg_temp'].interpolate()
            batch_sector_data['nitrit'] = batch_sector_data['nitrit'].interpolate()
            batch_sector_data['nitrate'] = batch_sector_data['nitrate'].interpolate()
            batch_sector_data['salt'] = batch_sector_data['salt'].interpolate()
            batch_sector_data['co2'] = batch_sector_data['co2'].interpolate()
            batch_sector_data['alkalinity'] = batch_sector_data['alkalinity'].interpolate()
            batch_sector_data['pH'] = batch_sector_data['pH'].interpolate()
        all_batch_data = all_batch_data.append(batch_sector_data, ignore_index=True)
    all_batch_data.fillna(method='bfill', inplace=True)
    return all_batch_data


def read_data(name, batch=True):
    all_data = pd.read_excel(cwd + '/' + name)
    if batch:
        return all_data
    # print(data.head())
    new_data = all_data[
        ["name", "date", "feed", "avg_temp", "ammonium", "nitrit", "nitrate", "salt", "co2", "alkalinity", "acidity",
         "pH", "density"]]
    # data_pavekst1 = new_data[new_data["name"] == "Påvekst 1"]
    # data_pavekst1.drop(columns=['pH', 'name'], inplace=True)
    # print(data_pavekst1.head())
    # print(len(data_pavekst1))
    # new_data.plot.scatter(x='date', y='feed', c=new_data['name'])4
    # print("dick")
    # sns.lmplot('id', 'feed', data=new_data, hue='name', col='name', fit_reg=False)
    # sns.PairGrid(data=data_pavekst1).map(sns.scatterplot)
    return new_data


if __name__ == '__main__':
    data = read_data('data_norway2.xlsx', batch=False)
    unique_locations = unique(data["name"])
    # plot_all_data(data, unique_locations, plot_batch=True)
    batches = all_batches_finder(data, unique_locations)
    # training_batch_data = (data, unique_locations, batches)
    training_batch_data = batch_data(data, unique_locations, batches, missing_data=True)
    print(training_batch_data.head())
    # plot_all_data(training_batch_data, unique_locations)
    training_batch_data = normalize_data(training_batch_data, save=False)
    save_data_by_name(training_batch_data, unique_locations)
    training_batch_data = one_hot_encode(training_batch_data, drop_date=True, ph_drop=True)
    # save_data(training_batch_data, 'batch_data3.xlsx')
