# =============================================================================
# Random Forest Regression
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# 3 months
data = data.loc[data.index > 2018070000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:15]
y = data.loc[:, 'Offers']

X.fillna(0, inplace = True)
y.fillna(0, inplace = True)

# small fix
X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 15% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.075, shuffle = False)

# feature scaling
sc_X = Normalizer(norm = 'l2')
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# empty list to append metric values
mae_gen = []
mse_gen  =[]
rmse_gen = []
mae_nor = []
mae_spi = []
mse_nor = []
mse_spi = []
rmse_nor = []
rmse_spi = []

from sklearn.ensemble import RandomForestRegressor

# create regressor 
regressor = RandomForestRegressor(n_estimators = 80)
regressor.fit(X_train, y_train)

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

results.to_csv('Results_Random_Forest.csv')

y_pred = y_pred.reshape(len(y_pred))

# 2 weeks = 14 (days) * 48 (SP) = 672 SP
Residual = list(y_test)[-672:] - y_pred[-672:]

plt.figure(figsize=(11,5))
plt.plot(y_pred[-672:], label = 'Predicted values', linewidth = 0.8)
plt.plot(list(y_test)[-672:], label = 'Real values', linewidth = 0.8)
plt.plot(Residual, label = 'Residual error', linewidth = 0.5)
a = len(y_pred[-672:])
#b = len(y_pred) - a
plt.xticks(np.arange(0, 700, 48), list(range(17, 32)))
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel(' Days of December 2018')
plt.ylabel('(£/MWh)')
plt.title('Random Forest: Real and predicted maximum accepted\n offer values for the last two weeks of 2018')
plt.legend()
plt.tight_layout()
plt.savefig('Random_Forest_Regression_prediction_without_FS.png')
