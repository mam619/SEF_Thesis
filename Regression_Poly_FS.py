# =============================================================================
# Multi- var Polynomial regression WITH Feature Selection
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)

# 2 months
data = data.loc[data.index > 2018070000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:15]
y = data.loc[:, 'Offers']

X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 15% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.075, shuffle=False)

# feature scaling
sc_X = MinMaxScaler()
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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# define polynomial regression degrees
degree = 2
poly_reg = PolynomialFeatures(degree = degree)
X_train = poly_reg.fit_transform(X_train)

# fit linear regression module to the matrix
regressor = LinearRegression()

# to append features chosen 
f_chosen_ = []

for i in range(1, len(X.columns)):
    
    # create feature selector
    selector = RFE(regressor, n_features_to_select = i, step = 1)
    
    # fit selector in data
    selector.fit(X_train, y_train)
    
    # features chosen
    a = X.columns.values
    f_chosen = a * selector.support_
    f_chosen = f_chosen[f_chosen != '']
    f_chosen_.append(f_chosen)

results_fs = pd.DataFrame({'n_features_to_select': (list(range(1, 15))),
                          
                           'features_chosen': f_chosen_})

for i in range(len(results_fs)):
    
    # Divide features and labels
    X = data.iloc[:, 0:15]
    y = data.loc[:, 'Offers']
    
    # select features
    X = X.loc[:, results_fs.loc[i][1]]

    X.fillna(X.mean(), inplace = True)
    y.fillna(y.mean(), inplace = True)

    X = X.astype('float64')
    X = X.round(20)
    
    # divide data into train and test with 15% test data
    X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.15, shuffle = False)
    
    # feature scaling
    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # create linear regressor 
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # make predictions with X_test
    y_pred = regressor.predict(poly_reg.fit_transform(X_test))
    
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
                    
                        'mae_normal': mae_nor} , index = results_fs)


results.to_csv('Results_Poly_Regression_FS.csv')



