# =============================================================================
# # =============================================================================
# # RNN structure - LSTM stateful vs. stateless 
# # =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# empty list to append metric values
mae_gen = []
mae_nor = []
mae_spi = []
rmse_gen = []
rmse_nor = []
rmse_spi = []

# =============================================================================
# import data & treat it
# =============================================================================
data = pd.read_csv('Data_set_1_smaller_(1).csv', index_col = 0)

# for later use
features_num = 14

# set predictive window according with tuning best results
data = data.loc[data.index > 2018070000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# fill nan values in the whole data set
data.fillna(method = 'ffill', inplace = True)

# =============================================================================
# train test split data
# =============================================================================
from sklearn.model_selection import train_test_split

# divide data into train and test 
data_train, data_test = train_test_split(
         data, test_size = 0.15, shuffle=False)

# =============================================================================
# do preprocessing no the wholde data set with division in train and test
# =============================================================================
from sklearn.preprocessing import MinMaxScaler

# data scaling  (including offer (y))
sc_X = MinMaxScaler()
data_train = sc_X.fit_transform(data_train)
data_test = sc_X.transform(data_test)

# =============================================================================
# funtions required for the LSTM structure
# =============================================================================
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

# =============================================================================
# parameters for LSTM
# =============================================================================
steps = 96
n_hidden = 1
units = 150 # did it with 50 before
batch_size = 96
epochs = 100

# =============================================================================
# set up X and y for train, test and val into correct shapes
# =============================================================================
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

# =============================================================================
# design the LSTM
# =============================================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import initializers
from keras import optimizers
from keras.callbacks import EarlyStopping

# STATELESS design of the LSTM
def regressor_tunning_stateless(kernel_initializer = 'he_normal',
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

# STATEFUL design of the LSTM
def regressor_tunning_stateful(kernel_initializer = 'he_normal',
                      bias_initializer = initializers.Ones()):
    model = Sequential()
    if n_hidden == 0:
        model.add(LSTM(units = units,                    
                       batch_input_shape = (batch_size, steps, features_num), 
                       stateful = True,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.2))
    else:
        model.add(LSTM(units = units,                    
                       batch_input_shape = (batch_size, steps, features_num), 
                       stateful = True,
                       return_sequences = True,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.2))
        model.add(LSTM(units = units, 
                       batch_input_shape = (batch_size, steps, features_num), 
                       stateful = True,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    optimizer = optimizers.RMSprop()
    model.compile(loss = 'mse', metrics = ['mse', 'mae'], optimizer = optimizer)
    return model

model_full = regressor_tunning_stateless()
model_less = regressor_tunning_stateful()

# apply patience callback
# early_stopping = EarlyStopping(monitor = 'val_mse', patience = 20)

# =============================================================================
# train the model
# =============================================================================

# fitting the LSTM to the training set
history_full = model_full.fit(X_train,
                         y_train, 
                         batch_size = batch_size, 
                         epochs = epochs,
                         shuffle = False, 
                         validation_data = (X_val, y_val))

# fitting the LSTM to the training set
history_less = model_less.fit(X_train,
                              y_train, 
                              batch_size = batch_size, 
                              epochs = epochs,
                              shuffle = False, 
                              validation_data = (X_val, y_val))

# =============================================================================
# create plots with rmse & mae during training
# =============================================================================
rmse = []
val_rmse = []

for i in history_full.history['mse']:
    rmse.append(i ** 0.5)
    
for i in history_full.history['val_mse']:
    val_rmse.append(i ** 0.5)
    
plt.figure(figsize=(9,4))
plt.plot(rmse, label = 'train')
plt.plot(val_rmse, label = 'test')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('RMSE(£/MWh)')
plt.tight_layout()
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title('RMSE during training for stateful LSTM')
plt.savefig('Plot_LSMT_stateful_RMSE_training_100_epochs.png')
plt.show()

plt.figure(figsize=(9,4))
plt.plot(history_full.history['mae'], label = 'train')
plt.plot(history_full.history['val_mae'], label = 'test')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('MAE(£/MWh)')
plt.tight_layout()
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title('MAE during training for stateful LSTM')
plt.savefig('Plot_LSMT_stateful_MAE_training_100_epochs.png')
plt.show()

for i in history_less.history['mse']:
    rmse.append(i ** 0.5)
    
for i in history_less.history['val_mse']:
    val_rmse.append(i ** 0.5)
    
plt.figure(figsize=(9,4))
plt.plot(rmse, label = 'train')
plt.plot(val_rmse, label = 'test')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('RMSE(£/MWh)')
plt.tight_layout()
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title('RMSE during training for stateless LSTM')
plt.savefig('Plot_LSMT_stateless_RMSE_training_100_epochs.png')
plt.show()

plt.figure(figsize=(9,4))
plt.plot(history_less.history['mae'], label = 'train')
plt.plot(history_less.history['val_mae'], label = 'test')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('MAE(£/MWh)')
plt.tight_layout()
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title('MAE during training for stateless LSTM')
plt.savefig('Plot_LSTM_stateless_MAE_training_100_epochs.png')
plt.show()

# =============================================================================
# make new predicitons with test set - STATEFULL first
# =============================================================================
y_pred = model_full.predict(X_test, batch_size = batch_size)

# =============================================================================
# y pred and y test required to inverse scaling
# =============================================================================
# cannot use inverse function; prices col = 14
y_pred = (y_pred * sc_X.data_range_[14]) + (sc_X.data_min_[14])
y_test = (y_test * sc_X.data_range_[14]) + (sc_X.data_min_[14])

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
# METRICS EVALUATION (2) on spike regions
# =============================================================================

# download spike indication binary set
y_spike_occ = pd.read_csv('Spike_binary_1std.csv', usecols = [6])

# create array same size as y_test
y_spike_occ = y_spike_occ.iloc[- len(y_test):]
y_spike_occ = pd.Series(y_spike_occ.iloc[:,0]).values

# smal adjustment
y_test = pd.Series(y_test)
y_test.replace(0, 0.0001,inplace = True)

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

# =============================================================================
# plot predictions for the end of 2018
# =============================================================================
# calculate residuals
Residual = list(y_test[-672:]) - y_pred[:,0][-672:]

# plot values
plt.figure(figsize=(11,5))
plt.plot(y_pred[-672:], label = 'Predicted values', linewidth = 0.8)
plt.plot(list(y_test)[-672:], label = 'Real values', linewidth = 0.8)
plt.plot(Residual, label = 'Residual error', linewidth = 0.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xticks(np.arange(0, 730, 48), list(range(17, 32)))
plt.xlabel(' Days of December 2018', fontsize = 12)
plt.ylabel('(£/MWh)', fontsize = 12)
plt.title('LSTM: Real and predicted maximum accepted offer values\n for the last two weeks of 2018'
          , fontsize=12)
plt.legend(loc = 'upper right')
plt.tight_layout()
#plt.savefig('Plot_LSTM_statless_prediction_500.png')

# plot for last day of 2018
Residual = list(y_test[-48:]) - y_pred[:,0][-48:]

# plot values
plt.figure(figsize=(11,5))
plt.plot(y_pred[-48:], label = 'Predicted values', linewidth = 0.8)
plt.plot(list(y_test)[-48:], label = 'Real values', linewidth = 0.8)
plt.plot(Residual, label = 'Residual error', linewidth = 0.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel(' Days of December 2018', fontsize = 12)
plt.ylabel('(£/MWh)', fontsize = 12)
plt.title('LSTM: Real and predicted maximum accepted offer values\n for the last two weeks of 2018'
          , fontsize=12)
plt.legend(loc = 'upper right')
plt.tight_layout()

# plot for last two days of 2018
Residual = list(y_test[-100:]) - y_pred[:,0][-100:]

# plot values
plt.figure(figsize=(11,5))
plt.plot(y_pred[-100:], label = 'Predicted values', linewidth = 0.8)
plt.plot(list(y_test)[-100:], label = 'Real values', linewidth = 0.8)
plt.plot(Residual, label = 'Residual error', linewidth = 0.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel(' Days of December 2018', fontsize = 12)
plt.ylabel('(£/MWh)', fontsize = 12)
plt.title('LSTM: Real and predicted maximum accepted offer values\n for the last two weeks of 2018'
          , fontsize=12)
plt.legend(loc = 'upper right')
plt.tight_layout()


# =============================================================================
# make new predicitons with test set - STATELESS now
# =============================================================================
y_pred = model_less.predict(X_test, batch_size = batch_size)

# =============================================================================
# y pred and y test required to inverse scaling
# =============================================================================
# cannot use inverse function; prices col = 14
y_pred = (y_pred * sc_X.data_range_[14]) + (sc_X.data_min_[14])
# y_test = (y_test * sc_X.data_range_[14]) + (sc_X.data_min_[14])

# =============================================================================
# METRICS EVALUATION (1) for the whole test set
# =============================================================================

# calculate metrics
rmse_error = mse(y_test, y_pred, squared = False)
mae_error = mae(y_test, y_pred)

# append to list
rmse_gen.append(rmse_error)
mae_gen.append(mae_error)

# =============================================================================
# METRICS EVALUATION (2) on spike regions
# =============================================================================

# download spike indication binary set
y_spike_occ = pd.read_csv('Spike_binary_1std.csv', usecols = [6])

# create array same size as y_test
y_spike_occ = y_spike_occ.iloc[- len(y_test):]
y_spike_occ = pd.Series(y_spike_occ.iloc[:,0]).values

# smal adjustment
y_test = pd.Series(y_test)
y_test.replace(0, 0.0001,inplace = True)

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

# =============================================================================
# plot predictions for the end of 2018
# =============================================================================
# calculate residuals
Residual = list(y_test[-672:]) - y_pred[:,0][-672:]

# plot values
plt.figure(figsize=(11,5))
plt.plot(y_pred[-672:], label = 'Predicted values', linewidth = 0.8)
plt.plot(list(y_test)[-672:], label = 'Real values', linewidth = 0.8)
plt.plot(Residual, label = 'Residual error', linewidth = 0.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xticks(np.arange(0, 730, 48), list(range(17, 32)))
plt.xlabel(' Days of December 2018', fontsize = 12)
plt.ylabel('(£/MWh)', fontsize = 12)
plt.title('LSTM: Real and predicted maximum accepted offer values\n for the last two weeks of 2018'
          , fontsize=12)
plt.legend(loc = 'upper right')
plt.tight_layout()
#plt.savefig('Plot_LSTM_statless_prediction_500.png')

# plot for last day of 2018
Residual = list(y_test[-48:]) - y_pred[:,0][-48:]

# plot values
plt.figure(figsize=(11,5))
plt.plot(y_pred[-48:], label = 'Predicted values', linewidth = 0.8)
plt.plot(list(y_test)[-48:], label = 'Real values', linewidth = 0.8)
plt.plot(Residual, label = 'Residual error', linewidth = 0.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel(' Days of December 2018', fontsize = 12)
plt.ylabel('(£/MWh)', fontsize = 12)
plt.title('LSTM: Real and predicted maximum accepted offer values\n for the last two weeks of 2018'
          , fontsize=12)
plt.legend(loc = 'upper right')
plt.tight_layout()

# plot for last two days of 2018
Residual = list(y_test[-100:]) - y_pred[:,0][-100:]

# plot values
plt.figure(figsize=(11,5))
plt.plot(y_pred[-100:], label = 'Predicted values', linewidth = 0.8)
plt.plot(list(y_test)[-100:], label = 'Real values', linewidth = 0.8)
plt.plot(Residual, label = 'Residual error', linewidth = 0.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel(' Days of December 2018', fontsize = 12)
plt.ylabel('(£/MWh)', fontsize = 12)
plt.title('LSTM: Real and predicted maximum accepted offer values\n for the last two weeks of 2018'
          , fontsize=12)
plt.legend(loc = 'upper right')
plt.tight_layout()

# =============================================================================
# save results
# =============================================================================

results = pd.DataFrame({'rmse_general': rmse_gen, 
                 
                        'mae_general': mae_gen,
                        
                        'rmse_spike': rmse_spi,
                 
                        'mae_spike': mae_spi,
                        
                        'rmse_normal': rmse_nor,
                    
                        'mae_normal': mae_nor}, index = ['Stateful', 'Stateless'])

results.to_csv('Results_LSTM_stateful_vs_stateful_100_epochs.csv')