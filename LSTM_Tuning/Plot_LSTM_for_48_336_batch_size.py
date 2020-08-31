# =============================================================================
# =============================================================================
# # Plot for comparison of performance of two LSTM structures 
# # with 48 SP batch size and steps - (1)
# # with 336 SP batch size and steps - (2)
# =============================================================================
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# downnload predictions
y_pred_336 = pd.read_csv('Prediction_LSTM_336_336.csv')
y_pred_48 = pd.read_csv('Prediction_LSTM_48_48.csv')

# =============================================================================
# Calculate y_test & spike limits for (1)
# =============================================================================

# parameters
steps = 336
batch_size = 336
features_num = 14

# months to evaluate model on
date = 2018110000

# import data
data = pd.read_csv('Data_set_1_smaller_(1).csv', index_col = 0)

# set predictive window according with tuning best results
data = data.loc[data.index > date, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# fill nan values in the whole data set
data.fillna(data.mean(), inplace = True)

from sklearn.model_selection import train_test_split

# divide data into train and test 
data_train, data_test = train_test_split(
         data, test_size = 0.15, shuffle=False)

from sklearn.preprocessing import MinMaxScaler

# data scaling  (including offer (y))
sc_X = MinMaxScaler()
data_train = sc_X.fit_transform(data_train)
data_test = sc_X.transform(data_test)


# function to split data into correct shape for RNN
def split_data(X, y, steps):
    X_, y_ = list(), list()
    for i in range(steps, len(y)):
        X_.append(X[i - steps : i, :])
        y_.append(y[i]) 
    return np.array(X_), np.array(y_)

    
# divide features and labels
X_train = data_train[:, 0:14] 
y_train = data_train[:, -1]
X_test = data_test[:, 0:14]
y_test = data_test[:, -1] 

# divide data into validation and normal test 
X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size = 0.15, shuffle=False)

# put data into correct shape
X_train, y_train = split_data(X_train, y_train, steps)
X_test, y_test = split_data(X_test, y_test, steps)
X_val, y_val = split_data(X_val, y_val, steps)

# y_test
y_test_336 = (y_test * sc_X.data_range_[-1]) + (sc_X.data_min_[-1])


# for spike limits:

# download data for shaded area
data = pd.read_csv('Spike_binary_1std.csv', index_col = 0)

# set predictive window according with tuning best results
data = data.loc[data.index > date, :]

# make sure shaded area will correspond to values outputed by LSTM
data.reset_index(drop = True, inplace = True)

# fill_nan is already made - so lets split data into test and train
from sklearn.model_selection import train_test_split

# divide data into train and test 
shade_train, shade_test = train_test_split(
         data, test_size = 0.15, shuffle = False)

# reset index of testing data
shade_test.reset_index(drop = True, inplace = True)

# function to split data into correct shape for RNN
def split_data(shade_test, steps):
    y_spike_occ = list()
    upper_lim = list()
    lower_lim = list()
    for i in range(steps, len(shade_test.index)):
        y_spike_occ.append(shade_test['spike_occurance'][i])
        upper_lim.append(shade_test['spike_upperlim'][i])
        lower_lim.append(shade_test['spike_lowerlim'][i])
    return np.array(y_spike_occ), np.array(upper_lim), np.array(lower_lim)

# function to cut data set so it can be divisible by the batch_size
def cut_data(data, batch_size):
     # see if it is divisivel
    condition = data.shape[0] % batch_size
    if condition == 0:
        return data
    else:
        return data[: -condition]

# shape y_spike_occ for the right size to compare results in normal and spike regions
y_spike_occ, spike_upperlim_336, spike_lowerlim_336 = split_data(shade_test, steps)

# =============================================================================
# Calculate y_test & spike limits for (2)
# =============================================================================

# parameters
steps = 48
batch_size = 48
features_num = 14

# months to evaluate model on
date = 2018110000

# import data
data = pd.read_csv('Data_set_1_smaller_(1).csv', index_col = 0)

# set predictive window according with tuning best results
data = data.loc[data.index > date, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# fill nan values in the whole data set
data.fillna(data.mean(), inplace = True)

from sklearn.model_selection import train_test_split

# divide data into train and test 
data_train, data_test = train_test_split(
         data, test_size = 0.15, shuffle=False)

from sklearn.preprocessing import MinMaxScaler

# data scaling  (including offer (y))
sc_X = MinMaxScaler()
data_train = sc_X.fit_transform(data_train)
data_test = sc_X.transform(data_test)


# function to split data into correct shape for RNN
def split_data(X, y, steps):
    X_, y_ = list(), list()
    for i in range(steps, len(y)):
        X_.append(X[i - steps : i, :])
        y_.append(y[i]) 
    return np.array(X_), np.array(y_)

    
# divide features and labels
X_train = data_train[:, 0:14] 
y_train = data_train[:, -1]
X_test = data_test[:, 0:14]
y_test = data_test[:, -1] 

# divide data into validation and normal test 
X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size = 0.15, shuffle=False)

# put data into correct shape
X_train, y_train = split_data(X_train, y_train, steps)
X_test, y_test = split_data(X_test, y_test, steps)
X_val, y_val = split_data(X_val, y_val, steps)

y_test_48 = (y_test * sc_X.data_range_[-1]) + (sc_X.data_min_[-1])


# for spike limits

# download data for shaded area
data = pd.read_csv('Spike_binary_1std.csv', index_col = 0)

# set predictive window according with tuning best results
data = data.loc[data.index > date, :]

# make sure shaded area will correspond to values outputed by LSTM
data.reset_index(drop = True, inplace = True)

# fill_nan is already made - so lets split data into test and train
from sklearn.model_selection import train_test_split

# divide data into train and test 
shade_train, shade_test = train_test_split(
         data, test_size = 0.15, shuffle = False)

# reset index of testing data
shade_test.reset_index(drop = True, inplace = True)

# function to split data into correct shape for RNN
def split_data(shade_test, steps):
    y_spike_occ = list()
    upper_lim = list()
    lower_lim = list()
    for i in range(steps, len(shade_test.index)):
        y_spike_occ.append(shade_test['spike_occurance'][i])
        upper_lim.append(shade_test['spike_upperlim'][i])
        lower_lim.append(shade_test['spike_lowerlim'][i])
    return np.array(y_spike_occ), np.array(upper_lim), np.array(lower_lim)

# function to cut data set so it can be divisible by the batch_size
def cut_data(data, batch_size):
     # see if it is divisivel
    condition = data.shape[0] % batch_size
    if condition == 0:
        return data
    else:
        return data[: -condition]

# shape y_spike_occ for the right size to compare results in normal and spike regions
y_spike_occ, spike_upperlim_48, spike_lowerlim_48 = split_data(shade_test, steps)


# =============================================================================
# Final plot with both
# =============================================================================

w_plot = 104 # 3 days
fontsize = 11

plt.figure(figsize=(9,3.5))

plt.subplot(2,1,1)
plt.plot(np.arange(0, (w_plot)), y_test_48[-w_plot:], label = 'Real values', linewidth = 1.5, color = 'steelblue')
plt.plot(np.arange(0, (w_plot)), y_pred_48.iloc[:, 1][-w_plot:], label = 'Predictions with 48 batch size', linewidth = 1.5, color= 'darksalmon')
#plt.plot(np.arange(0, (w_plot)), Residual[-w_plot:], label = 'Residual error', linewidth = 0.8, color = 'slategrey')
plt.fill_between(np.arange(0, (w_plot)),  spike_lowerlim_48[-w_plot:], spike_upperlim_48[-w_plot:], facecolor='skyblue', alpha=0.5, label = 'Spike delimitator')
plt.xlim(0, w_plot - 1)
#plt.ylim(-100, 260)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
##plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks([50,100, 150, 200, 250],[ 50, 100, 150, 200, 250],  fontsize = fontsize)
#plt.title('LSTM prediction using different optimizers', fontsize = fontsize + 2)
plt.legend(loc = 'upper right', framealpha = 0.5, fontsize = fontsize - 2)

plt.subplot(2,1,2)
plt.plot(np.arange(0, (w_plot)), y_test_336[-w_plot:], label = 'Real values', linewidth = 1.5, color = 'steelblue')
plt.plot(np.arange(0, (w_plot)), y_pred_336.iloc[:, 1][-w_plot:], label = 'Predictions with 336 batch size', linewidth = 1.5, color= 'darksalmon')
#plt.plot(np.arange(0, (w_plot)), Residual[-w_plot:], label = 'Residual error', linewidth = 0.8, color = 'slategrey')
plt.fill_between(np.arange(0, (w_plot)),  spike_lowerlim_336[-w_plot:], spike_upperlim_336[-w_plot:], facecolor='skyblue', alpha=0.5, label = 'Spike delimitator')
plt.xlim(0, w_plot - 1)
#plt.ylim(-100, 260)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks([50,100, 150, 200, 250],[ 50, 100, 150, 200, 250],  fontsize = fontsize)
#plt.title('LSTM prediction using different optimizers', fontsize = fontsize + 2)
plt.legend(loc = 'upper right', framealpha = 0.5, fontsize = fontsize - 2)
plt.tight_layout()

plt.savefig('Tunin1_48_336_plots.png')