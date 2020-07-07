import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import keras
# =============================================================================
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
# sess = tf.compat.v1.Session(config=config) 
# keras.backend.set_session(sess)
# =============================================================================

from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, BatchNormalization
from keras import Sequential

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# 2018 data
data = data.loc[data.index > 2018000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:15]
y = data.loc[:, 'Offers']

# Fill nan values (BEFORE OR AFTER TEST, TRAIN SPLIT!!!)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, shuffle=False)

# feature scaling aw
# Feature scalling to y ????????????????????????????????
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Hyperparameter tunning with LeakyRelu

def regressor_tunning(n_hidden = 2, n_neurons = 11, optimizer = 'RMSprop', kernel_initializer = 'he_normal'):
    model = Sequential()
    model.add(keras.layers.Dense(output_dim = n_neurons, kernel_initializer = 'he_normal', input_dim = 15))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    for layer in range(n_hidden):
        model.add(BatchNormalization())
        model.add(keras.layers.Dense(n_neurons, use_bias = False, kernel_initializer = 'he_normal'))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization())
    model.add(Dense(output_dim = 1, activation = 'linear',  use_bias = False, kernel_initializer = 'he_normal'))
    model.compile(loss = 'mse', metrics = ['mse', 'mae'], optimizer = optimizer)
    return model

def regressor_tunning_without(n_hidden = 2, n_neurons = 11, optimizer = 'RMSprop', kernel_initializer = 'he_normal'):
    model = Sequential()
    model.add(keras.layers.Dense(output_dim = n_neurons, kernel_initializer = 'he_normal', input_dim = 15))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, kernel_initializer = 'he_normal'))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dense(output_dim = 1, activation = 'linear', kernel_initializer = 'he_normal'))
    model.compile(loss = 'mse', metrics = ['mse', 'mae'], optimizer = optimizer)
    return model

regressor = KerasRegressor(build_fn = regressor_tunning)
regressor_without = KerasRegressor(build_fn = regressor_tunning_without)

tscv = TimeSeriesSplit(n_splits = 4)

hist_list = []
count = 1


for train_index, test_index in tscv.split(X_train):
      X_train_split, X_test_split = X_train[train_index], X_train[test_index]
      y_train_split, y_test_split = y_train[train_index], y_train[test_index]
      hist = regressor.fit(X_train_split, y_train_split, batch_size = 10, epochs = 40)
      hist_list.append(hist.history)
      print(count)
      count = count + 1
      

mse = []
loss = []
mae = []

for i in range(4):
    for j in range(40):
        mse.append(hist_list[i]['mse'][j])
        mae.append(hist_list[i]['mae'][j])
        
hist_list = []
      
for train_index, test_index in tscv.split(X_train):
      X_train_split, X_test_split = X_train[train_index], X_train[test_index]
      y_train_split, y_test_split = y_train[train_index], y_train[test_index]
      hist = regressor_without.fit(X_train_split, y_train_split, batch_size = 10, epochs = 40)
      hist_list.append(hist.history)
      print(count)
      count = count + 1

mse_without = []
mae_without = []


for i in range(4):
    for j in range(40):
        mse_without.append(hist_list[i]['mse'][j])
        mae_without.append(hist_list[i]['mae'][j])

# make them pretty  
fig = plt.figure(figsize = (16,3))

plt.subplot(1, 2, 1)
plt.plot(mse, label = 'with Batch Normalization')
plt.plot(mse_without)
plt.xlabel('Accumulated epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.title("Mean squared error during \n Nested Cross-Validation with 4 splits and 40 epochs")

plt.subplot(1, 2, 2)
plt.plot(mae, label = 'with Batch Normalization')
plt.plot(mae_without)
plt.xlabel('Accumulated epochs')
plt.ylabel('MAE (£/Mwh)')
plt.legend()
plt.grid()
plt.title("Mean absolute error during \nNested Cross-Validation with 4 splits and 40 epochs")
