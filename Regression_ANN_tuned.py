# =============================================================================
# Training and plotting of final ANN
# =============================================================================

import pandas as pd;
import numpy as np;
import sklearn
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict;
from sklearn.preprocessing import MinMaxScaler;
from sklearn import metrics;
from sklearn.model_selection import TimeSeriesSplit;

# empty list to append metric values
mae_cv = []
mse_cv = []
mae_gen = []
mse_gen  =[]
rmse_gen = []
mae_nor = []
mae_spi = []
mse_nor = []
mse_spi = []
rmse_nor = []
rmse_spi = []

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# 3 months
data = data.loc[data.index > 2018090000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:15]
y = data.loc[:, 'Offers']

X.fillna(method = 'ffill', inplace = True)
y.fillna(method = 'ffill', inplace = True)

X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.15, shuffle=False)

# feature scaling
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout
from keras import initializers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor

# possible debug
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

def regressor_tunning(n_hidden = 5, 
                      n_neurons = 40, 
                      kernel_initializer = "he_normal",
                      bias_initializer = initializers.Ones()):
    model = Sequential()
    model.add(Dense(units = n_neurons, input_dim = 15))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dropout(rate = 0.3))
    for layer in range(n_hidden):
        model.add(Dense(units = n_neurons))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
        model.add(Dropout(rate = 0.3))
    model.add(Dense(units = 1, activation = 'linear'))
    optimizer = optimizers.Adamax(lr = 0.001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mae'])
    return model

tscv = TimeSeriesSplit(n_splits = 10)

hist_list = pd.DataFrame()
count = 1
    
regressor = regressor_tunning()
    
for train_index, test_index in tscv.split(X_train):
    X_train_split, X_test_split = X_train[train_index], X_train[test_index]
    y_train_split, y_test_split = y_train[train_index], y_train[test_index]
    hist = regressor.fit(X_train_split, y_train_split,  shuffle = False, batch_size = 10, epochs = 100)
    hist_list = hist_list.append(hist.history, ignore_index = True)
    print(count)
    count = count + 1

a = []
b = []

for i in range(len(hist_list.mse)):
    a.append(np.mean(hist_list.mse[i]))
    b.append(np.mean(hist_list.mae[i]))

mse_cv.append(np.mean(a))
mae_cv.append(np.mean(b))

# predict for X_test  
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

rmse_error = mse(y_test, y_pred, squared = False)
mse_error = mse(y_test, y_pred) # 1479.61335
mae_error = mae(y_test, y_pred) # 23.1525

rmse_gen.append(rmse_error)
mse_gen.append(mse_error)
mae_gen.append(mae_error)

# =============================================================================
# Metrics evaluation on spike regions
# =============================================================================

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
mse_spike = mse(y_test_spike, y_pred_spike)
mae_spike = mae(y_test_spike, y_pred_spike)

rmse_spi.append(rmse_spike)
mse_spi.append(mse_spike)
mae_spi.append(mae_spike)

# =============================================================================
# Metric evaluation on normal regions
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
mse_normal = mse(y_test_normal, y_pred_normal)
mae_normal = mae(y_test_normal, y_pred_normal)

rmse_nor.append(rmse_normal)
mse_nor.append(mse_normal)
mae_nor.append(mae_normal)
# Save


results = pd.DataFrame({'rmse_general': rmse_gen, 
                 
                        'mae_general': mae_gen,
                        
                        'rmse_spike': rmse_spi,
                 
                        'mae_spike': mae_spi,
                        
                        'rmse_normal': rmse_nor,
                    
                        'mae_normal': mae_nor})

