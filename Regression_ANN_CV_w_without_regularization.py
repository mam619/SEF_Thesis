
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values

# 2017 & 2018 data
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


# ANN design

# importing the Keras libraries and packages
import keras

# =============================================================================
# # to run on gpu
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
# sess = tf.compat.v1.Session(config=config) 
# keras.backend.set_session(sess)
# =============================================================================

from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor

# =============================================================================
# # NO REGULARIZARTION
# =============================================================================

# initialises regressor with 2 hidden layers
regressor = Sequential() 
regressor.add(Dense(output_dim = 11, kernel_initializer = 'he_normal', input_dim = 15))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dense(11, kernel_initializer = 'he_normal'))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dense(11, kernel_initializer = 'he_normal'))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dense(output_dim = 1, kernel_initializer = 'he_normal', activation = 'linear'))
regressor.compile(optimizer = 'RMSprop', loss = 'mse', metrics = ['mse', 'mae'])

hist_list = pd.DataFrame()
tscv = TimeSeriesSplit(n_splits = 4)
count = 1

# do cv
for train_index, test_index in tscv.split(X_train):
      X_train_split, X_test_split = X_train[train_index], X_train[test_index]
      y_train_split, y_test_split = y_train[train_index], y_train[test_index]
      hist = regressor.fit(X_train_split, y_train_split, batch_size = 10, epochs = 40)
      hist_list = hist_list.append(hist.history, ignore_index = True)
      print(count)
      count = count + 1
      
# some inidividual instances:
mse_ = hist_list.mse
loss_ = hist_list.loss
mae_ = hist_list.mae

# append to then plot
mse = []
loss = []
mae = []

for i in range(4):
    for j in range(40):
        mse.append(mse_[i][j])
        loss.append(loss_[i][j])
        mae.append(mae_[i][j])


# =============================================================================
# # MODEL with regularization
# =============================================================================

# initialises regressor with 2 hidden layers
regressor = Sequential() 
regressor.add(Dense(output_dim = 11, kernel_initializer = 'he_normal', input_dim = 15))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dropout(p = 0.1))
regressor.add(Dense(11, kernel_initializer = 'he_normal'))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dropout(p = 0.1))
regressor.add(Dense(11,kernel_initializer = 'he_normal'))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dropout(p = 0.1))
regressor.add(Dense(output_dim = 1, activation = 'linear', kernel_initializer = 'he_normal'))
regressor.compile(optimizer = 'RMSprop', loss = 'mse', metrics = ['mse', 'mae'])

hist_list = pd.DataFrame()
tscv = TimeSeriesSplit(n_splits = 4)
count = 1

for train_index, test_index in tscv.split(X_train):
      X_train_split, X_test_split = X_train[train_index], X_train[test_index]
      y_train_split, y_test_split = y_train[train_index], y_train[test_index]
      hist = regressor.fit(X_train_split, y_train_split, batch_size = 10, epochs = 40)
      hist_list = hist_list.append(hist.history, ignore_index = True)
      print(count)
      count = count + 1
      
# some inidividual instances:
mse_ = hist_list.mse
loss_ = hist_list.loss
mae_ = hist_list.mae

dmse = []
dloss = []
dmae = []

for i in range(4):
    for j in range(40):
        dmse.append(mse_[i][j])
        dloss.append(loss_[i][j])
        dmae.append(mae_[i][j])

# =============================================================================
# # First plot with and without drop out
# =============================================================================
        
# make them pretty  
fig = plt.figure(figsize = (16,3))
plt.subplot(1, 2, 1)
plt.plot(mse)
plt.plot(dmse, label = 'with Dropout')
plt.xlabel('Accumulated epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.yticks(np.linspace(1000, 10000, 5))
plt.title("Mean squared error during \n Nested Cross-Validation with 4 splits and 40 epochs")
plt.subplot(1, 2, 2)
plt.plot(mae)
plt.plot(dmae, label = 'with Dropout')
plt.xlabel('Accumulated epochs')
plt.ylabel('MAE (£/Mwh)')
plt.yticks(np.linspace(20, 90, 5))
plt.grid()
plt.legend()
plt.title("Mean absolute error during \nNested Cross-Validation with 4 splits and 40 epochs")

# =============================================================================
# # Model with Batch Normalization
# =============================================================================

def regressor_tunning(n_hidden = 2, n_neurons = 11, optimizer = 'RMSprop'):
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

hist_list = []
count = 1


for train_index, test_index in tscv.split(X_train):
      X_train_split, X_test_split = X_train[train_index], X_train[test_index]
      y_train_split, y_test_split = y_train[train_index], y_train[test_index]
      hist = regressor.fit(X_train_split, y_train_split, batch_size = 10, epochs = 40)
      hist_list.append(hist.history)
      print(count)
      count = count + 1

mse_bn = []
mae_bn = []

for i in range(4):
    for j in range(40):
        mse_bn.append(hist_list[i]['mse'][j])
        mae_bn.append(hist_list[i]['mae'][j])
        
        
# =============================================================================
# Second plot with and without Batch Normalizations
# =============================================================================

# make them pretty  
fig = plt.figure(figsize = (16,3))

plt.subplot(1, 2, 1)
plt.plot(mse_bn, label = 'with Batch Normalization')
plt.plot(mse)
plt.xlabel('Accumulated epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.title("Mean squared error during \n Nested Cross-Validation with 4 splits and 40 epochs")

plt.subplot(1, 2, 2)
plt.plot(mae_bn, label = 'with Batch Normalization')
plt.plot(mae)
plt.xlabel('Accumulated epochs')
plt.ylabel('MAE (£/Mwh)')
plt.legend()
plt.grid()
plt.title("Mean absolute error during \nNested Cross-Validation with 4 splits and 40 epochs")


