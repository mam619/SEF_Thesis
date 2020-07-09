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

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values

# 2017 & 2018 data
data = data.loc[data.index > 2017000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:15]
y = data.loc[:, 'Offers']

X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, shuffle=False)

# feature scaling
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout
from keras import initializers
import keras.optimizers
from keras.wrappers.scikit_learn import KerasRegressor

# possible debug
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

def regressor_tunning(n_hidden = 2, 
                      n_neurons = 30, 
                      optimizer = 'Adamax', 
                      kernel_initializer = "he_normal",
                      bias_initializer = initializers.Ones()):
    model = Sequential()
    model.add(Dense(output_dim = n_neurons, input_dim = 15))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dropout(p = 0.1))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
        model.add(Dropout(p = 0.1))
    model.add(Dense(output_dim = 1, activation = 'linear'))
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mae'])
    return model

tscv = TimeSeriesSplit(n_splits = 11)

# DO CROSS VALIDATION
# TEST ON TRAINING SET
# FIND ERROR FOR BOTH ERROR AND NO ERROR SECTIONS
# DO PLOTS
# SEE HOW RESULTS CHANGE WITH DECREASE IN TRAINING DATA SET
