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
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# 2018 data
data = data.loc[data.index > 2018000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# fill nan values
data.fillna(method = 'ffill', inplace = True)

# divide features and labels
X_initial = data.iloc[:, 15].values # turns it into array
y_initial = data.loc[:, 'Offers'].values # turns it into array

#X = X.astype('float64')
#X = X.round(20)

def cut_data(X, y, steps):
    total = len(y)
    length = float(total/steps) * steps
    X = X[:length, :]
    y = y[:length]
    return X, y

def split_data(X, y, steps):
    X_, y_ = list(), list()
    for i in range(len(y)):
        # last index
        last_indx = i + steps
        # check if we are beyond dataset length
        if last_indx > len(y):
            break
        # gather X and y patterns
        X_.append(X[i:last_indx, :])
        y_.append(y[last_indx])
    return array(X_), array(y_)

steps = 96

X, y = cut_data(X_initial, y_initial, steps)
X, y = split_data(X_initial, y_initial, steps)

# divide data into train and test with 20% test data
# X_train, X_test, y_train, y_test = train_test_split(
#          X, y, test_size=0.2, shuffle=False)

# feature scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# LSTM design
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

def regressor_tunning(n_hidden = 1, n_neurons = 11, optimizer = 'adam'):
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (98,1)))
    model.add(Dropout(0.2))
    for layer in range(n_hidden):
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
    model.add(Dense(output_dim = 1))
    model.compile(loss = 'mse', metrics = ['mse', 'mae'], optimizer = optimizer)
    return model

# fit model and predict
tscv = TimeSeriesSplit(n_splits = 11)

LSTM_reg = KerasRegressor(build_fn = regressor_tunning, batch_size = 32, epochs = 100)
accuracies = cross_val_score(estimator = LSTM_reg, X = X_train, y = y_train, cv = tscv, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()
LSTM_reg.predict(X_test)