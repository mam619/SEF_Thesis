# =============================================================================
# =============================================================================
# # LSTM - tuned
# =============================================================================
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# empty list to append metric values
mae_gen = []
mae_nor = []
mae_spi = []
rmse_gen = []
rmse_nor = []
rmse_spi = []

# parameters
steps = 336
n_hidden = 2
units = 100
batch_size = 96
epochs = 200
features_num = 14

# months to evaluate model on
date = 2018090000

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

# function to cut data set so it can be divisible by the batch_size
def cut_data(data, batch_size):
     # see if it is divisivel
    condition = data.shape[0] % batch_size
    if condition == 0:
        return data
    else:
        return data[: -condition]
    
    
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

# cut data
X_train = cut_data(X_train, batch_size)
y_train = cut_data(y_train, batch_size)
X_test = cut_data(X_test, batch_size)
y_test = cut_data(y_test, batch_size)
X_val = cut_data(X_val, batch_size)
y_val = cut_data(y_val, batch_size)

# import libraries for LSTM design
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import initializers
from keras import optimizers

# function defining regressor
def regressor_tunning(kernel_initializer = 'he_uniform',
                      bias_initializer = initializers.Ones()):
    model = Sequential()
    if n_hidden == 0:
        model.add(LSTM(units = units,                    
                       input_shape = (steps, features_num), 
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.2))
    else:
        model.add(LSTM(units = units,                    
                       input_shape = (steps, features_num), 
                       return_sequences = True,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.2))
        model.add(LSTM(units = units, 
                       input_shape = (steps, features_num), 
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    optimizer = optimizers.RMSprop()
    model.compile(loss = 'mse', metrics = ['mse', 'mae'], optimizer = optimizer)
    return model

# create regressor
model = regressor_tunning()

history = model.fit(X_train,
                    y_train, 
                    batch_size = batch_size, 
                    epochs = epochs,
                    shuffle = False, 
                    validation_data = (X_val, y_val))
    
# required before predicitons
model.reset_states()

# make predictions with new data from X_test   
y_pred = model.predict(X_test, batch_size = batch_size)

# =============================================================================
# Evaluate predictions (3 stages)
# =============================================================================

# cannot use inverse function; prices col = 14
y_pred = (y_pred * sc_X.data_range_[-1]) + (sc_X.data_min_[-1])
y_test = (y_test * sc_X.data_range_[-1]) + (sc_X.data_min_[-1])

# Reshaping
y_pred = np.reshape(y_pred, (y_pred.shape[0]))

# =============================================================================
# METRICS EVALUATION (1) for the whole test set
# =============================================================================
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

# calculate metrics
rmse_error = mse(y_test, y_pred, squared = False)
mae_error = mae(y_test, y_pred)

# append to list
rmse_gen.append(rmse_error)
mae_gen.append(mae_error)

# =============================================================================
# (Need to process data with spike occurences in the same way)
# =============================================================================

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

y_spike_occ, spike_upperlim, spike_lowerlim = split_data(shade_test, steps)
y_spike_occ = cut_data(y_spike_occ, batch_size)
spike_upperlim = cut_data(spike_upperlim, batch_size)
spike_lowerlim = cut_data(spike_lowerlim, batch_size)

# =============================================================================
# METRICS EVALUATION (2) on spike regions
# =============================================================================

# smal adjustment
#y_test.replace(0, 0.0001,inplace = True)

# select y_pred and y_test only for regions with spikes
y_test_spike = (y_test.T * y_spike_occ).T
y_pred_spike = (y_pred.T * y_spike_occ).T
y_test_spike = y_test_spike[y_test_spike != 0]
y_pred_spike = y_pred_spike[y_pred_spike != 0]

# calculate metric
rmse_spike = mse(y_test_spike, y_pred_spike, squared = False)
mae_spike = mae(y_test_spike, y_pred_spike)

# append ot lists
rmse_spi.append(rmse_spike)
mae_spi.append(mae_spike)

# =============================================================================
# METRIC EVALUATION (3) on normal regions
# =============================================================================

# inverse y_spike_occ so the only normal occurences are chosen
y_normal_occ = (y_spike_occ - 1) * (-1)

# sanity check
y_normal_occ.sum() + y_spike_occ.sum() # gives the correct total 

# select y_pred and y_test only for normal regions
y_test_normal = (y_test.T * y_normal_occ).T
y_pred_normal = (y_pred.T * y_normal_occ).T
y_test_normal = y_test_normal[y_test_normal != 0.00]
y_pred_normal = y_pred_normal[y_pred_normal != 0.00]

# calculate metric
rmse_normal = mse(y_test_normal, y_pred_normal, squared = False)
mae_normal = mae(y_test_normal, y_pred_normal)

# append to list
rmse_nor.append(rmse_normal)
mae_nor.append(mae_normal)

# save results
results = pd.DataFrame({'rmse_general': rmse_gen, 
                 
                        'mae_general': mae_gen,
                        
                        'rmse_spike': rmse_spi,
                 
                        'mae_spike': mae_spi,
                        
                        'rmse_normal': rmse_nor,
                    
                        'mae_normal': mae_nor})

results.to_csv('Results_LSTM.csv')
print(results)

# =============================================================================
# Plot results for the end od 2018
# =============================================================================

w_plot = 144 # 3 days
fontsize = 13

y_pred = y_pred.reshape(len(y_pred))

Residual = list(y_test) - y_pred

plt.figure(figsize=(12.5,4))
plt.plot(np.arange(0, 144), y_test[-180:-36], label = 'Real values', linewidth = 2, color = 'steelblue')
plt.plot(np.arange(0, 144), y_pred[-180:-36], label = 'Predicted values', linewidth = 1.8, color= 'deepskyblue')
plt.plot(np.arange(0, 144), Residual[-180:-36], label = 'Residual error', linewidth = 1, color = 'slategrey')
plt.fill_between(np.arange(0, 144),  data['spike_lowerlim'][-180:-36], data['spike_upperlim'][-180:-36], facecolor='skyblue', alpha=0.5, label = 'Spike delimitator')
plt.ylim(-100, 260)
plt.xlim(0, 144 - 1)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks([-100, -50, 0, 50,100, 150, 200, 250],[-100, -50, 0, 50, 100, 150, 200, 250],  fontsize = fontsize)
plt.title('LSTM stateful predictions', fontsize = fontsize + 2)
plt.legend(loc = 'lower right', fontsize = fontsize - 2)
plt.tight_layout()
plt.savefig('Plot_LSTM_final.png')

y_pred = pd.Series(y_pred)
y_pred.to_csv('Prediction_LSTM.csv')

