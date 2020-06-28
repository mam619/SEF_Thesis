# =============================================================================
# ANN for Regression with Nested CV - 1 
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit

# import data
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values

# 2017 & 2018 data
data = data.loc[data.index > 2017000000, :]
data = data.loc[data.index < 2018123101, :]

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
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
'''
# initialises regressor with 2 hidden layers
regressor = Sequential() 
regressor.add(Dense(output_dim = 11, init = 'normal', activation = 'relu', input_dim = 20))
regressor.add(Dropout(p = 0.1))
regressor.add(Dense(output_dim = 11, init = 'normal', activation = 'relu'))
regressor.add(Dropout(p = 0.1))
regressor.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))

regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse', 'mae'])

hist = regressor.fit(X_train, y_train, batch_size = 10, epochs = 80, validation_split=0.2)

# =============================================================================
# plot
print(hist.history.keys())
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# from training last mae values around 28.7792; mse: 2147.2344 (rmse = 43.7)

# predicting from y_test:
y_pred = regressor.predict(X_test)

# metrics on test set:
mse = metrics.mean_squared_error(y_test, y_pred)# 2181.60
rmse = np.sqrt(mse) # 46.70
mae = metrics.mean_absolute_error(y_test, y_pred) # 26.48

# plot results
plt.plot(np.array(y_test), label = 'y_test', color = 'green', linewidth = 0.4)
plt.plot(y_pred, label = 'y_pred', color = 'red', linewidth = 0.4)
plt.ylabel ('£/MW')
plt.xlabel('Last 4 months of 2018')
plt.legend()
plt.title('Test set of Offers and correspondent predictions \nfor the last 4 months of 2018\n')
plt.show()

plt.plot(np.array(y_test)[0:48], label = 'y_test', color = 'green', linewidth = 0.4)
plt.plot(y_pred[0:48], label = 'y_pred', color = 'red', linewidth = 0.4)
plt.ylabel ('£/MW')
plt.xlabel('SP of 2018')
plt.legend()
plt.title('Test set of Offers and correspondent predictions \nfor the last 4 months of 2018\n')
plt.show()

plt.plot(np.array(y_test)[0:300], label = 'y_test', color = 'green', linewidth = 0.4)
plt.plot(y_pred[0:300], label = 'y_pred', color = 'red', linewidth = 0.4)
plt.ylabel ('£/MW')
plt.xlabel('SP of 2018')
plt.legend()
plt.title('Test set of Offers and correspondent predictions \nfor the last 4 months of 2018\n')
plt.show()


# Apply cross validation - cross_validate

def regressor_cv_():
    regressor = Sequential() 
    regressor.add(Dense(output_dim = 11, init = 'normal', activation = 'relu', input_dim = 20))
    regressor.add(Dropout(p = 0.1))
    regressor.add(Dense(output_dim = 11, init = 'normal', activation = 'relu'))
    regressor.add(Dropout(p = 0.1))
    regressor.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))
    regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse', 'mae'])
    return regressor

tscv = TimeSeriesSplit(n_splits = 11)
regressor_cv = KerasRegressor(build_fn = regressor_cv_, batch_size = 10, epochs = 100)
scores = cross_validate(regressor_cv, X_train, y_train, cv = tscv, scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error' ], n_jobs = -1)

# all values increase form previous calculus which makes sense!
scores['test_neg_mean_absolute_error'].mean() # 28.39
scores['test_neg_mean_absolute_error'].std() # 6.03
scores['test_neg_mean_squared_error'].mean() # 2300.97
scores['test_neg_mean_squared_error'].std() # 1701.54
# rmse mean 47.95
# rmse std 41

y_pred_1 = regressor_cv.predict(X_test)

# metrics on test set:
mse_cv = metrics.mean_squared_error(y_test, y_pred_1)# 2181.60
rmse_cv = np.sqrt(mse) # 46.70
mae_cv = metrics.mean_absolute_error(y_test, y_pred_1) # 26.48
'''
# =============================================================================
# # Apply cross validation - cross_val_predict
# 
# predictions = cross_val_predict(regressor_cv, X, y, cv = tscv, n_jobs = -1)
# 
# fig, ax = plt.subplots()
# ax.scatter(y, predictions, edgecolors=(0, 0, 0))
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# =============================================================================

# possible debug
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Add CV & Hyperparameter tunning with ReLu

def regressor_param(optimizer = 'adam', n_hidden = 1, n_neurons = 11, activation_fun = 'relu'):
    model = Sequential()
    model.add(keras.layers.Dense(output_dim = n_neurons, init = 'uniform', activation = activation_fun, input_dim = 20))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation = activation_fun))
    model.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))
    model.compile(loss = 'mse', optimizer = optimizer)
    return model

'''
# Add CV & Hyperparameter tunning with LeakyRelu

def regressor_param(optimizer = 'adam', n_hidden = 1, n_neurons = 11):
    model = Sequential()
    model.add(keras.layers.Dense(output_dim = n_neurons, init = 'uniform', input_dim = 20))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))
    model.compile(loss = 'mse', optimizer = optimizer)
    return model
'''

regressor = KerasRegressor(build_fn = regressor_param)

# Dictionary to include the parameters in GridSearch
parameters = {'n_hidden': [1, 2, 3, 4, 5, 6],
              }

tscv = TimeSeriesSplit(n_splits = 4)
n_iter_search = 20

rnd_search_cv = RandomizedSearchCV(estimator = regressor,
                                   param_distributions = parameters, 
                                   n_iter = n_iter_search, 
                                   scoring = 'neg_mean_squared_error',
                                   cv = tscv,
                                   error_score=0)
start = time.time()

early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10,
                                                  restore_best_weights = True)
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")

rnd_search_cv.fit(X_train, y_train, batch_size = 10, epochs = 20,
                  callbacks = [checkpoint_cb, early_stopping_cb])

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))

# FIND BEST PARAMETERS

# =============================================================================
# best_params = rnd_search_cv.best_params_
# best_score = rnd_search_cv.best_score_
# model_rnd_search = rnd_search_cv.best_estimator_.model
# =============================================================================

model = keras.models.load_model("my_keras_models.h5")