# =============================================================================
# # =============================================================================
# # Linear regression (multi var) WITH Feature Selection
# # =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE

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
data = data.loc[data.index > 2018110000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# =============================================================================
# Divide features and labels
# =============================================================================
X = data.iloc[:, 0:14]
y = data.loc[:, 'Offers']

X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

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
from sklearn.linear_model import LinearRegression

# create linear regressor 
regressor = LinearRegression()

# to append features chosen 
f_chosen_ = []

# =============================================================================
# Feature selection loop for different number of features to select
# =============================================================================
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

results_fs = pd.DataFrame({'n_features_to_select': (list(range(1, 14))),
                          
                           'features_chosen': f_chosen_})

# =============================================================================
# calculate results for every combination of features
# =============================================================================
for i in range(len(results_fs)):
    
    # Divide features and labels
    X = data.iloc[:, 0:14]
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
# Save results
# =============================================================================

results = pd.DataFrame({'rmse_general': rmse_gen, 
                 
                        'mae_general': mae_gen,
                        
                        'rmse_spike': rmse_spi,
                 
                        'mae_spike': mae_spi,
                        
                        'rmse_normal': rmse_nor,
                    
                        'mae_normal': mae_nor} , index = results_fs)


results.to_csv('Results_Linear_Regression_FS.csv')
