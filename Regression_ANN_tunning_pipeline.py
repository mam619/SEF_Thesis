# =============================================================================
# Hyperparameter tunning using pipeline
# 1) Pre processing data 
# =============================================================================

import pandas as pd
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, BatchNormalization()
from keras import Sequential

# import data
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# 2017 & 2018 data
data = data.loc[data.index > 2018000000, :]
data = data.loc[data.index < 2018090101, :]

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

# Hyperparameter tunning with LeakyRelu

def regressor_tunning(n_hidden = 1, n_neurons = 11, optimizer = 'adam'):
    model = Sequential()
    model.add(keras.layers.Dense(output_dim = n_neurons, init = 'uniform', input_dim = 20))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))
    model.compile(loss = 'mse', metrics = ['mse', 'mae'], optimizer = optimizer)
    return model

regressor = KerasRegressor(build_fn = regressor_tunning)

# Dictionary to include the parameters in GridSearch & RandomizedSearch
parameters_rand = {'n_hidden': [1, 2, 3, 4, 5, 6],
                   'n_neurons':[10, 20, 30, 40, 50, 60, 70, 80, 100]
                   }

parameters_grid = {'n_hidden': [1, 2, 3, 4, 5, 6]
                   }

tscv = TimeSeriesSplit(n_splits = 11)
n_iter_search = 20
start = time()

grd_search_cv = GridSearchCV(estimator = regressor,
                                   param_grid = parameters_grid, 
                                   scoring = 'neg_mean_squared_error',
                                   cv = tscv)

grd_search_cv.fit(X_train, y_train, batch_size = 10, epochs = 100)

print("GridSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))

# FIND BEST PARAMETERS
best_params = grd_search_cv.best_params_
print('Best parameters for grid search were: {}'.format(best_params))
best_score = grd_search_cv.best_score_
print('Best score in grid search was: {}'.format(best_score))

start = time()

rnd_search_cv = RandomizedSearchCV(estimator = regressor,
                                   param_distributions = parameters_rand, 
                                   n_iter = n_iter_search, 
                                   scoring = 'neg_mean_squared_error',
                                   cv = tscv,
                                   error_score=0)


rnd_search_cv.fit(X_train, y_train, batch_size = 10, epochs = 100)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))

# FIND BEST PARAMETERS
best_params = rnd_search_cv.best_params_
print('Best parameters for random search were: {}'.format(best_params))
best_score = rnd_search_cv.best_score_
print('Best score in grid random was: {}'.format(best_score))

