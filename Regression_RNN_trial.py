# =============================================================================
# RNN structure
# =============================================================================

import pandas as pd
import numpy as np

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
X = data.iloc[:, 0:15] .values # turns it into an array
y = data.loc[:, 'Offers'].values # turns it into an array

from sklearn.model_selection import train_test_split

# divide data into train and test 
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.15, shuffle=False)

from sklearn.preprocessing import MinMaxScaler

# feature scaling 
sc_X = MinMaxScaler()
X_train_ = sc_X.fit_transform(X_train)
X_test_ = sc_X.transform(X_test)

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

steps = 96

X_train, y_train = cut_data(X_train, y_train, steps)
X_test, y_test = cut_data(X_test, y_test, steps)

X_train, y_train = split_data(X_train, y_train, steps)
X_test, y_test = split_data(X_test, y_test, steps)


'''
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

'''