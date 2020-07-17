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

# empty list to append metric values
mae_cv = []
mse_cv = []
mae_gen = []
mse_gen  =[]
rmse_gen = []
mae_nor = []
mae_spi = []
mse_nor = []
mse_spi = []
rmse_nor = []
rmse_spi = []

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# 3 months
data = data.loc[data.index > 2018090000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:15]
y = data.loc[:, 'Offers']

X.fillna(method = 'ffill', inplace = True)
y.fillna(method = 'ffill', inplace = True)

X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.15, shuffle=False)

# feature scaling
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout
from keras import initializers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor

# possible debug
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

def regressor_tunning(n_hidden = 5, 
                      n_neurons = 40, 
                      kernel_initializer = "he_normal",
                      bias_initializer = initializers.Ones()):
    model = Sequential()
    model.add(Dense(units = n_neurons, input_dim = 15))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dropout(rate = 0.3))
    for layer in range(n_hidden):
        model.add(Dense(units = n_neurons))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
        model.add(Dropout(rate = 0.3))
    model.add(Dense(units = 1, activation = 'linear'))
    optimizer = optimizers.Adamax(lr = 0.001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mae'])
    return model

splits = 10 # 5
epochs = 100 # 80

tscv = TimeSeriesSplit(n_splits = splits)

hist_list = pd.DataFrame()
count = 1
    
regressor = regressor_tunning()

hist_list = pd.DataFrame()

for train_index, test_index in tscv.split(X_train):
    X_train_split, X_test_split = X_train[train_index], X_train[test_index]
    y_train_split, y_test_split = y_train[train_index], y_train[test_index]
    hist = regressor.fit(X_train_split, y_train_split,  
                         shuffle = False, 
                         validation_split = 0.2,
                         batch_size = 20, 
                         epochs = epochs)
    hist_list = hist_list.append(hist.history, ignore_index = True)
    print(count)
    count = count + 1

# plot train and test error during training (through epochs)
mse_ = hist_list.mse
mae_ = hist_list.mae
val_mse_ = hist_list.val_mse
val_mae_ = hist_list.val_mae

rmse = []
mae = []
val_rmse = []
val_mae = []

for i in range(splits):
    for j in range(epochs):
        rmse.append(mse_[i][j] ** 0.5)
        mae.append(mae_[i][j])
        val_rmse.append(val_mse_[i][j] ** 0.5)
        val_mae.append(val_mae_[i][j])

# make them pretty  
fig = plt.figure(figsize = (16,3))
plt.plot(rmse, label = 'train')
plt.plot(val_rmse, label = 'test')
plt.xlabel('Accumulated epochs')
plt.ylabel('RMSE (£/MWh)')
plt.grid()
#plt.yticks(np.linspace(1000, 18000, 5))
plt.title("RMSE for both training and validation \n during Nested Cross-Validation with {} splits".format(splits))
plt.show()


plt.plot(mae, label = 'train')
plt.plot(val_mae, label = 'test')
plt.xlabel('Accumulated epochs')
plt.ylabel('MAE (£/Mwh)')
plt.yticks(np.linspace(20, 120, 5))
plt.grid()
plt.title("MAE for both training and vbalidation \n during Nested Cross-Validation with {} splits".format(splits))

# predict for X_test  
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

# =============================================================================
# Metrics evaluation for the whole test set
# =============================================================================

rmse_error = mse(y_test, y_pred, squared = False)
mse_error = mse(y_test, y_pred) # 1479.61335
mae_error = mae(y_test, y_pred) # 23.1525

rmse_gen.append(rmse_error)
mse_gen.append(mse_error)
mae_gen.append(mae_error)

# =============================================================================
# Metrics evaluation on spike regions
# =============================================================================

y_spike_occ = pd.read_csv('Spike_binary_1std.csv', usecols = [6])

# create array same size as y_test
y_spike_occ = y_spike_occ.iloc[- len(y_test):]
y_spike_occ = pd.Series(y_spike_occ.iloc[:,0]).values

# smal adjustment
y_test.replace(0, 0.0001,inplace = True)

# select y_pred and y_test only for regions with spikes
y_test_spike = (y_test.T * y_spike_occ).T
y_pred_spike = (y_pred.T * y_spike_occ).T
y_test_spike = y_test_spike[y_test_spike != 0]
y_pred_spike = y_pred_spike[y_pred_spike != 0]

# calculate metric
rmse_spike = mse(y_test_spike, y_pred_spike, squared = False)
mse_spike = mse(y_test_spike, y_pred_spike)
mae_spike = mae(y_test_spike, y_pred_spike)

rmse_spi.append(rmse_spike)
mse_spi.append(mse_spike)
mae_spi.append(mae_spike)

# =============================================================================
# Metric evaluation on normal regions
# =============================================================================

# inverse y_spike_occ so the only normal occurences are chosen
y_normal_occ = (y_spike_occ - 1) * (-1)

# sanity check
y_normal_occ.sum() + y_spike_occ.sum() # gives the correct total 

# select y_pred and y_test only for normal regions
y_test_normal = (y_test.T * y_normal_occ).T
y_pred_normal = (y_pred.T * y_normal_occ).T
y_test_normal = y_test_normal[y_test_normal != 0.00]
y_pred_normal = y_pred_normal[y_pred_normal != 0.00]

# calculate metric
rmse_normal = mse(y_test_normal, y_pred_normal, squared = False)
mse_normal = mse(y_test_normal, y_pred_normal)
mae_normal = mae(y_test_normal, y_pred_normal)

rmse_nor.append(rmse_normal)
mse_nor.append(mse_normal)
mae_nor.append(mae_normal)

# Save
results = pd.DataFrame({'rmse_general': rmse_gen, 
                 
                        'mae_general': mae_gen,
                        
                        'rmse_spike': rmse_spi,
                 
                        'mae_spike': mae_spi,
                        
                        'rmse_normal': rmse_nor,
                    
                        'mae_normal': mae_nor})

y_pred = y_pred.reshape(len(y_pred))

Residual = list(y_test)[-672:] - y_pred[-672:]

plt.figure(figsize=(11,5))
plt.plot(y_pred[-672:], label = 'Predicted values', linewidth = 0.8)
plt.plot(list(y_test)[-672:], label = 'Real values', linewidth = 0.8)
plt.plot(Residual, label = 'Residual error', linewidth = 0.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xticks(np.arange(0, 730, 48), list(range(17, 32)))
plt.xlabel(' Days of December 2018')
plt.ylabel('(£/MWh)')
plt.title('Real and predicted maximum accepted offer values\n for the last two weeks of 2018')
plt.legend()
plt.tight_layout()
plt.savefig('ANN_prediction_without_FS.png')
