
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# 2018 data
data = data.loc[data.index > 2017000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# divide features and labels
X = data.iloc[:, 15]
y = data.loc[:, 'Offers']

X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.2, shuffle=False)


# 48 * 30 * 6
# 8640
# 8640 * 0.8
# 6912

tscv = TimeSeriesSplit(n_splits = 7, max_train_size = 6912)


for train_index, test_index in tscv.split(X_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_split, X_test_split = X_train[train_index], X_train[test_index]
    y_train_split, y_test_split = y_train[train_index], y_train[test_index]
    print(len(X_train_split) + len(X_test_split))
    
    
