# =============================================================================
# ANN for Regression with Nested CV - 1 
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit

# import data
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values


# 2017 & 2018 data
data = data.loc[data.index > 2017000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:21]
y = data.loc[:, 'Offers']

# Fill nan values (BEFORE OR AFTER TEST, TRAIN SPLIT!!!)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

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
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor

# initialises regressor with 2 hidden layers
regressor = Sequential() 
regressor.add(Dense(output_dim = 11, init = 'normal', activation = 'relu', input_dim = 21))
regressor.add(Dropout(p = 0.1))
regressor.add(Dense(output_dim = 11, init = 'normal', activation = 'relu'))
regressor.add(Dropout(p = 0.1))
regressor.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))

# compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse', 'mae'])

# fitting the ANN to the training set
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

# from training last mae values around 24.35; mse: 1910 (rmse = 43.7)

# predicting from y_test:
y_pred = regressor.predict(X_test)

# metrics on test set:
mse = metrics.mean_squared_error(y_test, y_pred)# 2566.43
rmse = np.sqrt(mse) # 50.66
mae = metrics.mean_absolute_error(y_test, y_pred) # 31.81

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

# Apply cross validation

tscv = TimeSeriesSplit(n_splits = 11)
scores = cross_val_score(regressor, X_train, y_train, cv = tscv, score = ['neg_mean_squared_error', 'neg_mean_absolute_error' ])

'''
# Add CV & Hyperparameterisation
def build_meg(n_hidden = 1, n_neuros = 30, learning_rate = ):
    regressor = Sequential() # Initialises
    regressor.add(Dense(output_dim = 11, init = 'normal', activation = 'relu', input_dim = 21))
    regressor.add(Dense(output_dim = 11, init = 'normal', activation = 'relu'))
    regressor.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))
    regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse', 'mae'])
    return regressor

def build_model(n_hidden = 1, n_neurons = 30, learning_rate = 3e-3, input_shape = [8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation = 'relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr = learning_rate)
    model.compile(loss = 'mse', optimizer = optimizer)
    return model

tscv = TimeSeriesSplit(n_splits=11)
regressor = KerasRegressor(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = regressor,X = X_train, y = y_train, cv = tscv, n_jobs = -1)
'''

