# =============================================================================
# # =============================================================================
# # Random Forest Regression
# # =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# empty list to append metric values
mae_gen = []
mae_nor = []
mae_spi = []
rmse_gen = []
rmse_nor = []
rmse_spi = []

# =============================================================================
# import data & treat it
# =============================================================================
data = pd.read_csv('Data_set_1_smaller_(1).csv', index_col = 0)

# set predictive window according with tuning best results
data = data.loc[data.index > 2018070000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# =============================================================================
# Divide features and labels
# =============================================================================
X = data.loc[:, ['APXP', 'TSDF', 'Im_Pr', 'PrevDay', 'DA_price_france']]
y = data.loc[:, 'Offers']

X.fillna(X.median(), inplace = True)
y.fillna(y.median(), inplace = True)

# small fix
X = X.astype('float64')
X = X.round(20)

# =============================================================================
# divide data into train and test
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.15, shuffle = False)

# =============================================================================
# feature scaling
# =============================================================================
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# =============================================================================
# regressor design 
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

# create regressor 
regressor = RandomForestRegressor(n_estimators = 80)
regressor.fit(X_train, y_train)

# =============================================================================
# predict for X_test  
# =============================================================================
y_pred = regressor.predict(X_test)

# =============================================================================
# METRICS EVALUATION (1) for the whole test set
# =============================================================================
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

# calculate metrics
rmse_error = mse(y_test, y_pred, squared = False)
mae_error = mae(y_test, y_pred)

# append to list
rmse_gen.append(rmse_error)
mae_gen.append(mae_error)

# =============================================================================
# METRICS EVALUATION (2) on spike regions
# =============================================================================

# download spike indication binary set
data = pd.read_csv('Spike_binary_1std.csv', index_col = 0)

y_spike_occ = data.spike_occurance

# create array same size as y_test
y_spike_occ = y_spike_occ.iloc[- len(y_test):]
y_spike_occ = y_spike_occ.values

# smal adjustment
y_test.replace(0, 0.0001,inplace = True)

# select y_pred and y_test only for regions with spikes
y_test_spike = (y_test.T * y_spike_occ).T
y_pred_spike = (y_pred.T * y_spike_occ).T
y_test_spike = y_test_spike[y_test_spike != 0]
y_pred_spike = y_pred_spike[y_pred_spike != 0]

# calculate metric
rmse_spike = mse(y_test_spike, y_pred_spike, squared = False)
mae_spike = mae(y_test_spike, y_pred_spike)

# append ot lists
rmse_spi.append(rmse_spike)
mae_spi.append(mae_spike)

# =============================================================================
# METRIC EVALUATION (3) on normal regions
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
mae_normal = mae(y_test_normal, y_pred_normal)

# append to list
rmse_nor.append(rmse_normal)
mae_nor.append(mae_normal)

# =============================================================================
# save results
# =============================================================================

results = pd.DataFrame({'rmse_general': rmse_gen, 
                 
                        'mae_general': mae_gen,
                        
                        'rmse_spike': rmse_spi,
                 
                        'mae_spike': mae_spi,
                        
                        'rmse_normal': rmse_nor,
                    
                        'mae_normal': mae_nor})


results.to_csv('Results_Random_Forest.csv')

# =============================================================================
# plot results for the end od 2018
# =============================================================================

w_plot = 144 # 3 days
fontsize = 13

y_pred = y_pred.reshape(len(y_pred))

Residual = list(y_test) - y_pred

plt.figure(figsize=(11,4))
plt.plot(np.arange(0, (w_plot)), y_test[-w_plot:], label = 'Real values', linewidth = 1.5, color = 'steelblue')
plt.plot(np.arange(0, (w_plot)), y_pred[-w_plot:], label = 'Predicted values', linewidth = 1.2, color= 'deepskyblue')
plt.plot(np.arange(0, (w_plot)), Residual[-w_plot:], label = 'Residual error', linewidth = 0.8, color = 'slategrey')
plt.fill_between(np.arange(0, (w_plot)),  data['spike_lowerlim'][-w_plot:],data['spike_upperlim'][-w_plot:], facecolor='skyblue', alpha=0.5, label = 'Not spike regions')
plt.xlim(0, w_plot - 1)
plt.ylim(-100, 260)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks([-100, -50, 0, 50,100, 150, 200, 250],[-100, -50, 0, 50, 100, 150, 200, 250],  fontsize = fontsize)
plt.title('Random Forest predictions', fontsize = fontsize + 2)
plt.legend(loc = 'lower right')
plt.tight_layout()
plt.savefig('Plot_Random_Forest_final.png')

y_pred = pd.Series(y_pred)
y_pred.to_csv('Prediction_Random_Forest.csv')


