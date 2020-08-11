# =============================================================================
# ANN for Regression with Nested CV - 1 & plot
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit

# import data
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values

# 2017 & 2018 data
data = data.loc[data.index > 2018000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:20]
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

# to run on gpu
import tensorflow as tf
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)

from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor

# initialises regressor with 2 hidden layers
regressor = Sequential() 
regressor.add(Dense(output_dim = 11, init = 'uniform', input_dim = 20))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dense(11))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dense(11))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dense(11))
regressor.add(keras.layers.LeakyReLU(alpha = 0.2))
regressor.add(Dense(output_dim = 1, activation = 'linear'))
regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse', 'mae'])

hist_list = pd.DataFrame()
tscv = TimeSeriesSplit(n_splits = 11)
count = 1

for train_index, test_index in tscv.split(X_train):
      X_train_split, X_test_split = X_train[train_index], X_train[test_index]
      y_train_split, y_test_split = y_train[train_index], y_train[test_index]
      hist = regressor.fit(X_train_split, y_train_split, batch_size = 10, epochs = 100)
      hist_list = hist_list.append(hist.history, ignore_index = True)
      print(count)
      count = count + 1
      
# Plot the loss/mse/mae per epoch
mse_ = hist_list.mse
loss_ = hist_list.loss
mae_ = hist_list.mae

mse = []
loss = []
mae = []

for i in range(11):
    for j in range(100):
        mse.append(mse_[i][j])
        loss.append(loss_[i][j])
        mae.append(mae_[i][j])

# make them pretty  
fig = plt.figure(figsize = (16,3))
plt.subplot(1, 2, 1)
plt.plot(mse)
plt.xlabel('Accumulated epochs')
plt.ylabel('MSE')
plt.grid()
plt.yticks(np.linspace(1000, 18000, 5))
plt.title("Mean squared error during \n Nested Cross-Validation with 11 splits")
plt.subplot(1, 2, 2)
plt.plot(mae)
plt.xlabel('Accumulated epochs')
plt.ylabel('MAE (£/Mwh)')
plt.yticks(np.linspace(20, 120, 5))
plt.grid()
plt.title("Mean absolute error during \nNested Cross-Validation with 11 splits")

# some inidividual instances:
fig = plt.figure(figsize = (8,4))
plt.plot(mse[100:200])
plt.xlabel('Accumulated epochs')
plt.ylabel('MSE')
plt.title('Second epoch')

fig = plt.figure(figsize = (8,4))
plt.plot(mae[100:200])
plt.xlabel('Accumulated epochs')
plt.ylabel('MAE (£/Mwh')
plt.title('Second epoch')

fig = plt.figure(figsize = (8,4))
plt.plot(mse[300:400])
plt.xlabel('Accumulated epochs')
plt.ylabel('MSE')
plt.title('Fourth epoch')

fig = plt.figure(figsize = (8,4))
plt.plot(mae[300:400])
plt.xlabel('Accumulated epochs')
plt.ylabel('MAE (£/Mwh')
plt.title('Frouth epoch')

