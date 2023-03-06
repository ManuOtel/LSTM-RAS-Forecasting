"""RAS Digitalization project.

This module contains the script that is handle the data that is going to.

@Author: Emanuel-Ionut Otel
@Company: Billund Aquaculture A/S
@Created: 2022-02-07
@Contact: manuotel@gmail.com
"""


#### ---- IMPORTS AREA ---- ####
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from pickle import dump
from typing import List
#### ---- IMPORTS AREA ---- ####


#### ---- GLOBAL INIT AREA ---- ####
cwd = os.getcwd()
# cwd = cwd + '/data'
#### ---- GLOBAL INIT AREA ---- ####


def save_data(save_to_data:pd.DataFrame, filename:str) -> None:
    """
    Save the data frame to a file

    param: save_to_data: pd.DataFrame The data frame to be saved
    param: filename:     str The name of the file to be saved to

    return: None
    """
    save_to_data.to_excel(filename)


def one_hot_encode(normal_data:pd.DataFrame, drop_date:bool=False, ph_drop:bool=False) -> pd.DataFrame:
    """
    One hot encode the name column of the data frame
    
    param: normal_data:  pd.DataFrame The data frame to be encoded
    param: drop_date:    bool Drop the date column
    param: ph_drop:      bool Drop the pH column
    
    return:              pd.DataFrame The encoded data frame
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_normal_data = pd.DataFrame(encoder.fit_transform(normal_data[['name']]).toarray())
    encoded_data = normal_data.join(encoder_normal_data)
    encoded_data.drop('name', axis=1, inplace=True)
    if drop_date:
        encoded_data.drop('date', axis=1, inplace=True)
    if ph_drop:
        encoded_data.drop('pH', axis=1, inplace=True)
    return encoded_data


def normalize_data(full_data:pd.DataFrame, save:bool=True) -> pd.DataFrame:
    """
    Normalize the data frame values

    param: full_data:    pd.DataFrame The data frame to be normalized
    param: save:         bool Save the scaler (default: True)

    return:              pd.DataFrame The normalized data frame
    """
    aux_data = full_data.select_dtypes(include='number')
    normalized_data = pd.DataFrame()
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    for i in aux_data:
        x = aux_data[i].values.reshape(1, -1).astype(np.float)  # returns a numpy array
        x_scaled = min_max_scaler.fit_transform(np.transpose(x)).astype(np.float)
        normalized_data[i] = x_scaled.reshape(-1, ).astype(np.float)
        if save:
            dump(min_max_scaler, open('scaler_' + str(i) + '.pkl', 'wb'))
    full_data[normalized_data.columns] = normalized_data
    return full_data


def unique(initial_list:List) -> List:
    """
    Return a list with unique values from the initial list

    param: initial_list: list The list to be filtered

    return:              list The filtered list
    """
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in initial_list:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def batch_finder(sector_data:pd.DataFrame) -> List:
    """
    Find the indexes of the batches inside the data frame

    param: sector_data:  pd.DataFrame The data frame to be filtered

    return:              list The list of batch indexes
    """
    batch_index = []
    last_value = 0  # sector_data["density"][1]
    for idx in sector_data.index:
        i = sector_data["density"][idx]
        der = abs(i - last_value)
        if der > 0:
            batch_index.append(idx)
    return batch_index


def plot_single_data(plot_data:pd.DataFrame, figure_name:str, plot_batch:bool=False) -> None:
    """
    Plots the data frame

    param: plot_data:    pd.DataFrame The data frame to be plotted
    param: figure_name:  str The name of the figure
    param: plot_batch:   bool Plot the batches

    return:              None
    """
    # fig = plt.figure()
    plt.subplots(constrained_layout=True)

    plt.subplot(811).set_title('Density', loc='right', fontsize=10)
    plt.plot(range(len(plot_data.index)), plot_data["density"])
    plt.title(figure_name)

    if plot_batch:
        batch_index = batch_finder(plot_data)
        for i in batch_index:
            plt.axvline(x=i, c='red')
    # print(batch_index)

    plt.subplot(812).set_title('Feed', loc='right', fontsize=10)
    plt.plot(range(len(plot_data.index)), plot_data["feed"], 'r')
    plt.scatter(range(len(plot_data.index)), plot_data["feed"], c='r')

    plt.subplot(813).set_title('Salt', loc='right', fontsize=10)
    plt.plot(range(len(plot_data.index)), plot_data["salt"])
    plt.scatter(range(len(plot_data.index)), plot_data["salt"])

    plt.subplot(814).set_title('Temp', loc='right', fontsize=10)
    plt.plot(range(len(plot_data.index)), plot_data["avg_temp"])
    plt.scatter(range(len(plot_data.index)), plot_data["avg_temp"])

    plt.subplot(815).set_title('Ammonium', loc='right', fontsize=10)
    plt.plot(range(len(plot_data.index)), plot_data["ammonium"])
    plt.scatter(range(len(plot_data.index)), plot_data["ammonium"])

    plt.subplot(816).set_title('Nitrit', loc='right', fontsize=10)
    plt.plot(range(len(plot_data.index)), plot_data["nitrit"])
    plt.scatter(range(len(plot_data.index)), plot_data["nitrit"])

    plt.subplot(817).set_title('Nitrate', loc='right', fontsize=10)
    plt.plot(range(len(plot_data.index)), plot_data["nitrate"])
    plt.scatter(range(len(plot_data.index)), plot_data["nitrate"])

    plt.subplot(818).set_title('CO2', loc='right', fontsize=10)
    plt.plot(range(len(plot_data.index)), plot_data["co2"])
    plt.scatter(range(len(plot_data.index)), plot_data["co2"])

    # out = plt.get_figure()
    # print(out)

    plt.show()
    # return fig, ax


def plot_all_data(total_data:pd.DataFrame, unique_name:List, plot_batch:bool=False) -> None:
    """
    Plots all the data frames

    param: total_data:   pd.DataFrame The data frame to be plotted
    param: unique_name:  List The list of unique names
    param: plot_batch:   bool Plot the batches (default: False)

    return:              None
    """
    for i in unique_name:
        plot_single_data(total_data[total_data["name"] == i], i, plot_batch)



def all_batches_finder(total_data:pd.DataFrame, unique_name:List) -> List:
    """
    Find the indexes of the batches inside the data frame

    param: total_data:   pd.DataFrame The data frame to be filtered
    param: unique_name:  List The list of unique names

    return:              List The list of batch indexes
    """
    all_batches = [None] * len(unique_name)
    for i in unique_name:
        # print(total_data[total_data["name"] == i])
        current_batch = batch_finder(total_data[total_data["name"] == i])
        all_batches[unique_name.index(i)] = current_batch
    return all_batches


def save_data_by_name(total_data:pd.DataFrame, unique_name:List) -> None:
    """
    Save the data frames by name

    param: total_data:   pd.DataFrame The data frame to be filtered
    param: unique_name:  List The list of unique names

    return:              None
    """
    for i in unique_name:
        named_data = total_data[total_data['name'] == i]
        save_data(named_data, 'named_data/' + i + '_data.xlsx')


def batch_data(total_data:pd.DataFrame, unique_name:List, all_batches:List, missing_data:bool=False) -> pd.DataFrame:
    """
    Create a data frame with the batches

    param: total_data:   pd.DataFrame The data frame to be filtered
    param: unique_name:  List The list of unique names
    param: all_batches:  List The list of batch indexes
    param: missing_data: bool Interpolate the missing data (default: False)

    return:              pd.DataFrame The data frame with the batches
    """
    all_batch_data = pd.DataFrame()
    for i in unique_name:
        sector_data = total_data[total_data["name"] == i]
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


def read_data(name:str, batch:bool=True) -> pd.DataFrame:
    """
    Read the data from the excel file

    param: name:  str The name of the file
    param: batch: bool Read the whole data (default: True)

    return:       pd.DataFrame The data frame with the data
    """
    all_data = pd.read_excel(name)
    if batch:
        return all_data
    new_data = all_data[["name", 
                         "date", 
                         "feed", 
                         "avg_temp", 
                         "ammonium", 
                         "nitrit", 
                         "nitrate", 
                         "salt", 
                         "co2", 
                         "alkalinity", 
                         "acidity",
                         "pH", 
                         "density"]]
    return new_data


if __name__ == '__main__':
    data = read_data('data_norway2.xlsx', batch=False)
    unique_locations = unique(data["name"])
    batches = all_batches_finder(data, unique_locations)
    training_batch_data = batch_data(data, unique_locations, batches, missing_data=True)
    print(training_batch_data.head())
    plot_all_data(training_batch_data, unique_locations)
    training_batch_data = normalize_data(training_batch_data, save=False)
    save_data_by_name(training_batch_data, unique_locations)
    training_batch_data = one_hot_encode(training_batch_data, drop_date=True, ph_drop=True)
