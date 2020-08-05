# =============================================================================
# # =============================================================================
# # ANN
# # =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# empty list to append metric values
mse_cv = []
mae_cv = []
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

# set predictive window according with tuning best results
data = data.loc[data.index > 2018110000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# =============================================================================
# Divide features and labels
# =============================================================================
X = data.iloc[:, 0:14]
y = data.loc[:, 'Offers']

X.fillna(X.median(), inplace = True)
y.fillna(y.median(), inplace = True)

X = X.astype('float64')
X = X.round(20)

# =============================================================================
# divide data into train and test
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.15, shuffle = False)

# =============================================================================
# feature scaling
# =============================================================================
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# =============================================================================
# regressor design 
# =============================================================================
import keras
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout
from keras import initializers
from keras import optimizers

# possible debug
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

def regressor_tunning(n_hidden = 4, 
                      n_neurons = 20, 
                      kernel_initializer = "he_normal",
                      bias_initializer = initializers.Ones()):
    model = Sequential()
    model.add(Dense(units = n_neurons, input_dim = 14))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dropout(rate = 0.5))
    for layer in range(n_hidden):
        model.add(Dense(units = n_neurons))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
        model.add(Dropout(rate = 0.5))
    model.add(Dense(units = 1, activation = 'linear'))
    optimizer = optimizers.Adamax(lr = 0.0001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mae'])
    return model


# =============================================================================
# parameters for the ANN
# =============================================================================
splits = 13 
epochs = 180

# =============================================================================
# prepare CV splits, count and data frame to append results during training
# =============================================================================
tscv = TimeSeriesSplit(n_splits = splits)

hist_list = pd.DataFrame()
count = 1
    
regressor = regressor_tunning()

hist_list = pd.DataFrame()

# =============================================================================
# train model
# =============================================================================

for train_index, test_index in tscv.split(X_train):
    X_train_split, X_test_split = X_train[train_index], X_train[test_index]
    y_train_split, y_test_split = y_train[train_index], y_train[test_index]
    hist = regressor.fit(X_train_split, y_train_split,  
                         shuffle = False, 
                         validation_split = 0.2,
                         batch_size = 20, 
                         epochs = epochs)
    hist_list = hist_list.append(hist.history, ignore_index = True)
    print(count)
    count = count + 1

# =============================================================================
# train and test error during training (through epochs)
# =============================================================================
mse_ = hist_list.mse
mae_ = hist_list.mae
val_mse_ = hist_list.val_mse
val_mae_ = hist_list.val_mae

rmse = []
mae = []
val_rmse = []
val_mae = []

for i in range(splits):
    for j in range(epochs):
        rmse.append(mse_[i][j] ** 0.5)
        mae.append(mae_[i][j])
        val_rmse.append(val_mse_[i][j] ** 0.5)
        val_mae.append(val_mae_[i][j])

# =============================================================================
# plot errors during training (RMSE & MAE)        
# =============================================================================
import matplotlib.pyplot as plt

# make them pretty  
fig = plt.figure(figsize = (16,4))
plt.plot(rmse, label = 'train')
plt.plot(val_rmse, label = 'test')
plt.xlabel('Accumulated epochs', fontsize = 13)
plt.ylabel('RMSE (£/MWh)', fontsize = 13)
plt.legend(fontsize = 13)
plt.minorticks_on()
plt.yticks(np.linspace(20, 120, 5), fontsize = 13)
plt.xticks(fontsize = 13)
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
#plt.yticks(np.linspace(1000, 18000, 5))
plt.title("RMSE during Nested Cross-Validation\n with {} splits for 4 months of data ".format(splits), fontsize = 14)
plt.savefig('Plot_ANN_RMSE_during_training.png')
plt.show()

fig = plt.figure(figsize = (16,4))
plt.plot(mae, label = 'train')
plt.plot(val_mae, label = 'test')
plt.legend(fontsize = 13)
plt.xlabel('Accumulated epochs', fontsize = 13)
plt.ylabel('MAE (£/Mwh)', fontsize = 13)
plt.yticks(np.linspace(20, 120, 5), fontsize = 13)
plt.xticks(fontsize = 13)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title("MAE during Nested Cross-Validation\n with {} splits for 4 months of data".format(splits), fontsize = 14)
plt.savefig('Plot_ANN_MAE_during_training.png')
plt.show()

# =============================================================================
# predict for X_test  
# =============================================================================
y_pred = regressor.predict(X_test)

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
# save results
# =============================================================================

results = pd.DataFrame({'rmse_general': rmse_gen, 
                 
                        'mae_general': mae_gen,
                        
                        'rmse_spike': rmse_spi,
                 
                        'mae_spike': mae_spi,
                        
                        'rmse_normal': rmse_nor,
                    
                        'mae_normal': mae_nor})

results.to_csv('Best_Results_ANN_normal.csv')

