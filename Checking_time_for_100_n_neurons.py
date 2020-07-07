import pandas as pd
import numpy as np
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

import keras
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# possible debug
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Add CV & Hyperparameter tunning with ReLu

def regressor_tunning(n_hidden = 1, n_neurons = 11, optimizer = 'adam', kernel_initializer="glorot_uniform",
    bias_initializer="zeros"):
    model = Sequential()
    model.add(keras.layers.Dense(output_dim = n_neurons, init = 'uniform', input_dim = 20))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mae'])
    return model

regressor = KerasRegressor(build_fn = regressor_tunning)

# Dictionary to include the parameters
parameters = {'n_hidden': [5],
              'n_neurons': [100]
               }

tscv = TimeSeriesSplit(n_splits = 7)

# add some early stopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='mse', patience = 15)

rnd_search_cv = RandomizedSearchCV(estimator = regressor,
                                   param_distributions = parameters,
                                   scoring = 'neg_mean_squared_error',
                                   n_iter = 20,
                                   cv = tscv)

rnd_search_cv.fit(X_train, y_train, batch_size = 10, epochs = 60, callbacks=[early_stopping])