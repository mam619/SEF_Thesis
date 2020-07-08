# =============================================================================
# First RNN structure
# =============================================================================

# data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# import data
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# 2018 data
data = data.loc[data.index > 2018060000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# divide features and labels
X = data.iloc[:, 0:20]
y = data.loc[:, 'Offers']

X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, shuffle=False)

# feature scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# LSTM design
import keras
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

def regressor_tunning(n_hidden = 1, n_neurons = 11, optimizer = 'adam'):
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    for layer in range(n_hidden):
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
    model.add(Dense(output_dim = 1))
    model.compile(loss = 'mse', metrics = ['mse', 'mae'], optimizer = optimizer)
    return model

# fit model and predict
tscv = TimeSeriesSplit(n_splits = 11)

LSTM_reg = KerasRegressor(build_fn = regressor_tunning, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = LSTM_reg, X = X_train, y = y_train, cv = tscv, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()
LSTM_reg.predict(X_test)