# =============================================================================
# Benchmark models: Naive (Current SP), Naive (last day SP), Naive (Last Week SP), Mean values 
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# import data
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max())

# shift offers 3 SP back
data['Offers'] = data['Offers'].shift(-3)

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
         X, y, test_size = 0.2, shuffle=False)

# =============================================================================
# Mean Benchmark
# =============================================================================

# y.mean() 112.59
# y_train.mean() 111.97

y_pred_mean = np.ones(len(y_test)) * y_train.mean()

rmse_mean = metrics.mean_squared_error(y_test, y_pred_mean, squared = False)
# 46.24
mse_mean = metrics.mean_squared_error(y_test, y_pred_mean)
# 2138.84
mae_mean = metrics.mean_absolute_error(y_test, y_pred_mean)
# 25.48

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test)),y_test, linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test)), y_pred_mean - y_test, linewidth = 1, label = 'Residual Error')
plt.plot(np.arange(0, len(y_test)),y_pred_mean, linewidth = 1.5, label = 'Constant value prediction', color = 'black')
plt.legend()
plt.xlabel('Last 4 months of 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with constant value \nprediction (using trainning set mean value)\n Residual error also included ')
plt.show()

# =============================================================================
# Naive Benchmark - Most recent SP
# =============================================================================

data_1 = pd.read_csv('Data_set_1.csv', index_col = 0)

data_1['Offers'].fillna(data_1['Offers'].mean(), inplace = True)

y_naive_1 = data_1['Offers'].shift(1)
y_naive_1 = y_naive_1[-len(y_test) :]
y_test = data_1['Offers'].shift(-3)
y_test = y_test[-len(y_naive_1) :]
y_test.fillna(y_train.mean(), inplace = True)

rmse_naive1 = metrics.mean_squared_error(y_test, y_naive_1, squared = False)
print(rmse_naive1)
# 59.61
mse_naive1 = metrics.mean_squared_error(y_test, y_naive_1)
print(mse_naive1)
# 3553.23
mae_naive1 = metrics.mean_absolute_error(y_test, y_naive_1)
print(mae_naive1)
# 27.99

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test))[0:48],y_test[0:48], linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test))[0:48],y_naive_1[0:48], linewidth = 1, label = 'Most resent SP prediction', color = 'orange')
plt.plot(np.arange(0, len(y_test))[0:48], (y_naive_1 - y_test)[0:48], linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Last 4 months of 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with most recent SP period as prediction \n Residual error also included ')
plt.show()

# =============================================================================
# Naive Benchmark - Previous Day same SP
# =============================================================================

data = pd.read_csv('Data_set_1.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max())

# shift offers 3 SP back
data['Offers'] = data['Offers'].shift(-3)

# 2017 & 2018 data
data = data.loc[data.index > 2017000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:21]
y = data.loc[:, 'Offers']

X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.2, shuffle=False)

y_naive_2 = X_test['PrevDay']

rmse_naive2 = metrics.mean_squared_error(y_test, y_naive_2, squared = False)
print(rmse_naive2)
# 62.35
mse_naive2 = metrics.mean_squared_error(y_test, y_naive_2)
print(mse_naive2)
# 3887.93
mae_naive2 = metrics.mean_absolute_error(y_test, y_naive_2)
print(mae_naive2)
# 22.82

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test))[0:48],y_test[0:48], linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test))[0:48],y_naive_2[0:48], linewidth = 1, label = 'Same SP form last day', color = 'orange')
plt.plot(np.arange(0, len(y_test))[0:48], (y_naive_2 - y_test)[0:48], linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Full day from 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with same SP from previous day \n Residual error also included ')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test)),y_test, linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test)),y_naive_2, linewidth = 1, label = 'Same SP form last day', color = 'orange')
plt.plot(np.arange(0, len(y_test)), (y_naive_2 - y_test), linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Last 4 months of 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with same SP from previous day \n Residual error also included ')
plt.show()


# =============================================================================
# Naive Benchmark - Previous Week same SP
# =============================================================================

data = pd.read_csv('Data_set_1.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max())

# shift offers 3 SP back
data['Offers'] = data['Offers'].shift(-3)

# 2017 & 2018 data
data = data.loc[data.index > 2017000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:21]
y = data.loc[:, 'Offers']

X.fillna(X.mean(), inplace = True)
y.fillna(y.mean(), inplace = True)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.2, shuffle=False)

y_naive_3 = X_test['PrevWeek']

rmse_naive3 = metrics.mean_squared_error(y_test, y_naive_3, squared = False)
print(rmse_naive3)
# 63.54
mse_naive3 = metrics.mean_squared_error(y_test, y_naive_3)
print(mse_naive3)
# 4037.42
mae_naive3 = metrics.mean_absolute_error(y_test, y_naive_3)
print(mae_naive3)
# 33.91

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test))[0:48],y_test[0:48], linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test))[0:48],y_naive_3[0:48], linewidth = 1, label = 'Same SP form last week', color = 'orange')
plt.plot(np.arange(0, len(y_test))[0:48], (y_naive_3 - y_test)[0:48], linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Full day from 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with same SP from previous week \n Residual error also included ')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test)),y_test, linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test)),y_naive_3, linewidth = 1, label = 'Same SP form last week', color = 'orange')
plt.plot(np.arange(0, len(y_test)), (y_naive_3 - y_test), linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Last 4 months of 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with same SP from previous week \n Residual error also included ')
plt.show()

rmse = [rmse_mean, rmse_naive1, rmse_naive2, rmse_naive3 ]
mae = [mae_mean, mae_naive1, mae_naive2, mae_naive3 ]
results = pd.DataFrame({'rmse': rmse, 'mae':mae})
results['index'] = ['Mean_benchmark', 'Naive_1', 'Naive_2', 'Naive_3']
results.set_index('index', inplace = True)
results = results.T