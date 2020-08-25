# =============================================================================
# Multi- var linear regression //Predictive window tuning
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# empty list to append metric values
rmse_gen = []
mae_gen = []

rmse_nor = []
mae_nor = []

rmse_spi = []
mae_spi = []


dates = [2017010000,
         2017030000, 
         2017050000, 
         2017070000,
         2017090000, 
         2017110000,
         2018010000,
         2018030000, 
         2018050000,
         2018070000,
         2018090000,
         2018110000]

dates_labels = ['24 ', 
                '22 ',
                '20 ',  
                '18 ', 
                '16 ', 
                '14 ', 
                '12 ',
                '10 ',
                '8 ',
                '6 ',
                '4 ',
                '2 ']

degree = 2

for i in dates:
    
    # import data
    data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)
    
    # 3 months
    data = data.loc[data.index > i, :]
    
    # reset index
    data.reset_index(inplace = True)
    data.drop('index', axis = 1, inplace = True)
    
    # Divide features and labels
    X = data.iloc[:, 0:15]
    y = data.loc[:, 'Offers']
    
    # fill nan values
    X.fillna(X.mean(), inplace = True)
    y.fillna(y.mean(), inplace = True)
    
    # small fix
    X = X.astype('float64')
    X = X.round(20)
    
    # divide data into train and test with 15% test data
    X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.075, shuffle = False)
    
    # feature scaling
    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    # define polynomial regression degrees
    poly_reg = PolynomialFeatures(degree = degree)
    X_train = poly_reg.fit_transform(X_train)
    
    # fit linear regression module to the matrix
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # predict for X_test  
    y_pred = regressor.predict(poly_reg.fit_transform(X_test))
    
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    
    # =============================================================================
    # Metrics evaluation for the whole test set
    # =============================================================================
    
    rmse_error = mse(y_test, y_pred, squared = False)
    mae_error = mae(y_test, y_pred) # 23.1525
    
    rmse_gen.append(rmse_error)
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
    mae_spike = mae(y_test_spike, y_pred_spike)
    
    rmse_spi.append(rmse_spike)
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
    mae_normal = mae(y_test_normal, y_pred_normal)
    
    rmse_nor.append(rmse_normal)
    mae_nor.append(mae_normal)

# Save
results = pd.DataFrame({'rmse_general': rmse_gen, 
                 
                        'mae_general': mae_gen,
                        
                        'rmse_spike': rmse_spi,
                 
                        'mae_spike': mae_spi,
                        
                        'rmse_normal': rmse_nor,
                    
                        'mae_normal': mae_nor})

results.to_csv('Results_Polynomial_Regression_Predictive_window.csv')

fontsize = 13

plt.figure(figsize=(10,3.5))
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title('Polynomial Regression: RMSE for different training time window sizes', fontsize = fontsize + 2)
plt.plot(rmse_gen, label = 'All test set')
plt.plot(rmse_spi, label = 'Spike regions')
plt.plot(rmse_nor, label = 'Non - spike regions')
plt.xlim(0, 11)
plt.legend(loc = 'upper right', fontsize = fontsize - 2)
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xlabel('Time window (in months)', fontsize = fontsize)
plt.xticks(list(range(len(rmse_spi))), dates_labels, fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.tight_layout()
plt.savefig('RMSE_predictive_window.png')
