# =============================================================================
# RNN structure
# =============================================================================

import pandas as pd
import numpy as np

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# for later use
features_num = 15

# 2018 data
data = data.loc[data.index > 2018000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# fill nan values
data.fillna(method = 'ffill', inplace = True)

# divide features and labels
X = data.iloc[:, 0:15] .values # turns it into an array
y = data.loc[:, 'Offers'].values # turns it into an array

from sklearn.model_selection import train_test_split

# divide data into train and test 
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.15, shuffle=False)

from sklearn.preprocessing import MinMaxScaler

# feature scaling 
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#X = X.astype('float64')
#X = X.round(20)

# function to cut data
def cut_data(X, y, steps):
    total = len(y)
    length = int(total/steps) * steps
    X = X[:length, :]
    y = y[:length]
    return X, y

# function to split data into correct shape
def split_data(X, y, steps):
    X_, y_ = list(), list()
    for i in range(steps, len(y)):
        X_.append(X[i - steps : i, :])
        y_.append(y[i]) 
    return np.array(X_), np.array(y_)

steps = 10

X_train, y_train = cut_data(X_train, y_train, steps)
X_test, y_test = cut_data(X_test, y_test, steps)

X_train, y_train = split_data(X_train, y_train, steps)
X_test, y_test = split_data(X_test, y_test, steps)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import initializers
from keras import optimizers

# parameters
n_hidden = 1
units = 10

# design the LSTM
model = Sequential()
model.add(LSTM(units = units, 
               return_sequences = True,
               input_shape = (X_train.shape[1], features_num),
               kernel_initializer = 'he_normal',
               bias_initializer = initializers.Ones()))
model.add(LSTM(units = units))
# model.add(keras.layers.LeakyReLU(aplha = 0.2))
model.add(Dropout(0.2))
# =============================================================================
# for layer in range(n_hidden):
#     model.add(LSTM(units = units, 
#                    return_sequences = True,
#                    kernel_initializer = 'he_normal',
#                    bias_initializer = initializers.Ones()))
# # model.add(keras.layers.LeakyReLU(aplha = 0.2))
#     model.add(Dropout(0.2))
# =============================================================================
model.add(Dense(1, activation='linear'))
optimizer = optimizers.Adamax(lr = 0.001)
model.compile(loss = 'mse', metrics = ['mse', 'mae'], optimizer = optimizer)

# fitting the RNN to the Training set
model.fit(X_train, y_train, batch_size = 2, epochs = 4)

y_pred = model.predict(X_test)

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
y_test = pd.Series(y_test)
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

