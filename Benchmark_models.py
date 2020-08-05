# =============================================================================
# =============================================================================
# Benchmark models: 
# # Mean values 
# # Naive (Current SP)
# # Naive (last day SP)
# # Naive (Last Week SP)
# =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# =============================================================================
# import data // no need to shift data as it has already been done
# =============================================================================
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# 2018 data
data = data.loc[data.index > 2018000000, :]

# reset index
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)

# drop nan values
data.dropna(inplace = True)

# Divide features and labels
X = data.iloc[:, 0:21]
y = data.loc[:, 'Offers']

# divide data into train and test with 15% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.15, shuffle=False)

w = 48
# create individual data set for binary spike creation with droped nan values
# create the moving average data set with correspondent rolling window
data['sma'] = data['Offers'].rolling(window = w).mean()
# create the standard deviation data set with correspondent rolling window
data['std'] = data['Offers'].rolling(w).std()
# create upper and lower limits data set
data['spike_upperlim'] = data['sma'] + (data['std'])
data['spike_lowerlim'] = data['sma'] - (data['std'])
# create binary data set with occurences out of these limis
data['spike_occurance'] = ((data['Offers'] > data['spike_upperlim']) | (data['Offers'] < data['spike_lowerlim'])).astype(np.int)

# =============================================================================
# Mean Benchmark - general
# =============================================================================

# y.mean() 112.59
# y_train.mean() 111.97

# create y_test for mean benchmark
y_pred_mean = np.ones(len(y_test)) * y_train.mean()

# create array to divide spike and normal region
y_spike_occ = data.iloc[- len(y_test):,-1]

# total error of this benchmark
rmse_mean = metrics.mean_squared_error(y_test, y_pred_mean, squared = False)
# 46.24
mse_mean = metrics.mean_squared_error(y_test, y_pred_mean)
# 2138.84
mae_mean = metrics.mean_absolute_error(y_test, y_pred_mean)
# 25.48

print("For Mean benchmark rmse: {}, mae: {} and mse: {}".format(rmse_mean, mae_mean, mse_mean))

# =============================================================================
# Mean Benchmark - spike regions
# =============================================================================

# smal adjustment
y_test = y_test.where(y_test > 0)
y_test.fillna('0.01', inplace = True)
y_test = y_test.astype('float64')

# select y_pred and y_test only for regions with spikes
y_test_spike = y_test * y_spike_occ
y_mean_spike = y_pred_mean * y_spike_occ
y_test_spike = y_test_spike[y_test_spike != 0.00]
y_mean_spike = y_mean_spike[y_mean_spike != 0.00]

# calculate metric
rmse_mean_spike = metrics.mean_squared_error(y_test_spike, y_mean_spike, squared = False)
mse_mean_spike = metrics.mean_squared_error(y_test_spike, y_mean_spike)
mae_mean_spike = metrics.mean_absolute_error(y_test_spike, y_mean_spike)

print("For Mean benchmark on spike regions rmse: {}, mae: {} and mse: {}".format(rmse_mean_spike, mae_mean_spike, mse_mean_spike))

# =============================================================================
# Mean Benchmark - normal regions
# =============================================================================

# inverse y_spike_occ so the only normal occurences are chosen
y_normal_occ = (y_spike_occ - 1) * (-1)

# sanity check
y_normal_occ.sum() + y_spike_occ.sum() # gives the correct total of 6774

# select y_pred and y_test only for normal regions
y_test_normal = y_test * y_normal_occ
y_mean_normal = y_pred_mean * y_normal_occ
y_test_normal = y_test_normal[y_test_normal != 0.00]
y_mean_normal = y_mean_normal[y_mean_normal != 0.00]

# calculate metric
rmse_mean_normal = metrics.mean_squared_error(y_test_normal, y_mean_normal, squared = False)
mse_mean_normal = metrics.mean_squared_error(y_test_normal, y_mean_normal)
mae_mean_normal = metrics.mean_absolute_error(y_test_normal, y_mean_normal)

print("For Mean benchmark on normal regions rmse: {}, mae: {} and mse: {}".format(rmse_mean_normal, mae_mean_normal, mse_mean_normal))

# =============================================================================
# Naive Benchmark (Most recent SP) - general
# =============================================================================

y_naive1 = data['Offers'].shift(1)
y_naive1 = y_naive1[-len(y_test) :]

rmse_naive1 = metrics.mean_squared_error(y_test, y_naive1, squared = False)
mse_naive1 = metrics.mean_squared_error(y_test, y_naive1)
mae_naive1 = metrics.mean_absolute_error(y_test, y_naive1)

print("For Naive 1 benchmark rmse: {}, mae: {} and mse: {}".format(rmse_naive1, mae_naive1, mse_naive1))

# =============================================================================
# Naive 1 - spike regions
# =============================================================================

# select y_pred and y_test only for regions with spikes
y_test_spike = y_test * y_spike_occ
y_naiv1_spike = y_naive1 * y_spike_occ
y_test_spike = y_test_spike[y_test_spike != 0.00]
y_naiv1_spike = y_naiv1_spike[y_naiv1_spike != 0.00]

# calculate metric
rmse_naive1_spike = metrics.mean_squared_error(y_test_spike, y_naiv1_spike, squared = False)
mse_naive1_spike = metrics.mean_squared_error(y_test_spike, y_naiv1_spike)
mae_naive1_spike = metrics.mean_absolute_error(y_test_spike, y_naiv1_spike)

print("For Naive 1 benchmark on spike regions rmse: {}, mae: {} and mse: {}".format(rmse_naive1_spike, mae_naive1_spike, mse_naive1_spike))

# =============================================================================
# Naive 1 - normal regions
# =============================================================================

# smal adjustment
y_naive1 = y_naive1.where(y_naive1 > 0)
y_naive1.fillna(0.01, inplace = True)

# select y_pred and y_test only for normal regions
y_test_normal = y_test * y_normal_occ
y_naive1_normal = y_naive1 * y_normal_occ

y_test_normal = y_test_normal[y_test_normal != 0.000]
y_naive1_normal = y_naive1_normal[y_naive1_normal != 0.000]

# calculate metric
rmse_naive1_normal = metrics.mean_squared_error(y_test_normal, y_naive1_normal, squared = False)
mse_naive1_normal = metrics.mean_squared_error(y_test_normal, y_naive1_normal)
mae_naive1_normal = metrics.mean_absolute_error(y_test_normal, y_naive1_normal)

print("For Naive 1 benchmark on normal regions rmse: {}, mae: {} and mse: {}".format(rmse_naive1_normal, mae_naive1_normal, mse_naive1_normal))

# some plotting
plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test))[0:48],y_test[0:48], linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test))[0:48],y_naive1[0:48], linewidth = 1, label = 'Most resent SP prediction', color = 'orange')
plt.plot(np.arange(0, len(y_test))[0:48], (y_naive1 - y_test)[0:48], linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Last 4 months of 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with most recent SP period as prediction \n Residual error also included ')
plt.show()

# =============================================================================
# Naive Benchmark (Previous Day same SP) - general
# =============================================================================

y_naive2 = X_test['PrevDay']

rmse_naive2 = metrics.mean_squared_error(y_test, y_naive2, squared = False)
mse_naive2 = metrics.mean_squared_error(y_test, y_naive2)
mae_naive2 = metrics.mean_absolute_error(y_test, y_naive2)

print("For Naive 2 benchmark rmse: {}, mae: {} and mse: {}".format(rmse_naive2, mae_naive2, mse_naive2))

# =============================================================================
# Naive 2 - spike regions
# =============================================================================

# select y_pred and y_test only for regions with spikes
y_test_spike = y_test * y_spike_occ
y_naive2_spike = y_naive2 * y_spike_occ
y_test_spike = y_test_spike[y_test_spike != 0.00]
y_naive2_spike = y_naive2_spike[y_naive2_spike != 0.00]

# calculate metric
rmse_naive2_spike = metrics.mean_squared_error(y_test_spike, y_naive2_spike, squared = False)
mse_naive2_spike = metrics.mean_squared_error(y_test_spike, y_naive2_spike)
mae_naive2_spike = metrics.mean_absolute_error(y_test_spike, y_naive2_spike)

print("For Naive 2 benchmark on spike regions rmse: {}, mae: {} and mse: {}".format(rmse_naive2_spike, mae_naive2_spike, mse_naive2_spike))

# =============================================================================
# Naive 2 - normal regions
# =============================================================================

# smal adjustment
y_naive2 = y_naive2.where(y_naive2 > 0)
y_naive2.fillna(0.01, inplace = True)

# select y_pred and y_test only for normal regions
y_test_normal = y_test * y_normal_occ
y_naive2_normal = y_naive2 * y_normal_occ

y_test_normal = y_test_normal[y_test_normal != 0.000]
y_naive2_normal = y_naive2_normal[y_naive2_normal != 0.000]

# calculate metric
rmse_naive2_normal = metrics.mean_squared_error(y_test_normal, y_naive2_normal, squared = False)
mse_naive2_normal = metrics.mean_squared_error(y_test_normal, y_naive2_normal)
mae_naive2_normal = metrics.mean_absolute_error(y_test_normal, y_naive2_normal)

print("For Naive 2 benchmark on normal regions rmse: {}, mae: {} and mse: {}".format(rmse_naive2_normal, mae_naive2_normal, mse_naive2_normal))


# some plotting
plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test))[0:48],y_test[0:48], linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test))[0:48],y_naive2[0:48], linewidth = 1, label = 'Same SP form last day', color = 'orange')
plt.plot(np.arange(0, len(y_test))[0:48], (y_naive2 - y_test)[0:48], linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Full day from 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with same SP from previous day \n Residual error also included ')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test)),y_test, linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test)),y_naive2, linewidth = 1, label = 'Same SP form last day', color = 'orange')
plt.plot(np.arange(0, len(y_test)), (y_naive2 - y_test), linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Last 4 months of 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with same SP from previous day \n Residual error also included ')
plt.show()


# =============================================================================
# Naive Benchmark (Previous Week same SP) - general
# =============================================================================

y_naive3 = X_test['PrevWeek']

rmse_naive3 = metrics.mean_squared_error(y_test, y_naive3, squared = False)
mse_naive3 = metrics.mean_squared_error(y_test, y_naive3)
mae_naive3 = metrics.mean_absolute_error(y_test, y_naive3)

print("For Naive 3 benchmark rmse: {}, mae: {} and mse: {}".format(rmse_naive3, mae_naive3, mse_naive3))

# =============================================================================
# Naive 3 - spike regions
# =============================================================================

# select y_pred and y_test only for regions with spikes
y_test_spike = y_test * y_spike_occ
y_naive3_spike = y_naive3 * y_spike_occ
y_test_spike = y_test_spike[y_test_spike != 0.00]
y_naive3_spike = y_naive3_spike[y_naive3_spike != 0.00]

# calculate metric
rmse_naive3_spike = metrics.mean_squared_error(y_test_spike, y_naive3_spike, squared = False)
mse_naive3_spike = metrics.mean_squared_error(y_test_spike, y_naive3_spike)
mae_naive3_spike = metrics.mean_absolute_error(y_test_spike, y_naive3_spike)

print("For Naive 3 benchmark on spike regions rmse: {}, mae: {} and mse: {}".format(rmse_naive3_spike, mae_naive3_spike, mse_naive3_spike))

# =============================================================================
# Naive 3 - normal regions
# =============================================================================

# smal adjustment
y_naive3 = y_naive3.where(y_naive3 > 0)
y_naive3.fillna(0.01, inplace = True)

# select y_pred and y_test only for normal regions
y_test_normal = y_test * y_normal_occ
y_naive3_normal = y_naive3 * y_normal_occ

y_test_normal = y_test_normal[y_test_normal != 0.000]
y_naive3_normal = y_naive3_normal[y_naive3_normal != 0.000]

# calculate metric
rmse_naive3_normal = metrics.mean_squared_error(y_test_normal, y_naive3_normal, squared = False)
mse_naive3_normal = metrics.mean_squared_error(y_test_normal, y_naive3_normal)
mae_naive3_normal = metrics.mean_absolute_error(y_test_normal, y_naive3_normal)

print("For Naive 3 benchmark on normal regions rmse: {}, mae: {} and mse: {}".format(rmse_naive3_normal, mae_naive3_normal, mse_naive3_normal))


# some plotting
plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test))[0:48],y_test[0:48], linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test))[0:48],y_naive3[0:48], linewidth = 1, label = 'Same SP form last week', color = 'orange')
plt.plot(np.arange(0, len(y_test))[0:48], (y_naive3 - y_test)[0:48], linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Full day from 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with same SP from previous week \n Residual error also included ')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(y_test)),y_test, linewidth = 1, label = 'Real value', color = 'green')
plt.plot(np.arange(0, len(y_test)),y_naive3, linewidth = 1, label = 'Same SP form last week', color = 'orange')
plt.plot(np.arange(0, len(y_test)), (y_naive3 - y_test), linewidth = 1, label = 'Residual Error')
plt.legend()
plt.xlabel('Last 4 months of 2018')
plt.ylabel('£/MWh')
plt.title('Plotting of true values with same SP from previous week \n Residual error also included ')
plt.show()

rmse = [rmse_mean, rmse_naive1, rmse_naive2, rmse_naive3]
mae = [mae_mean, mae_naive1, mae_naive2, mae_naive3]
mse = [mse_mean, mse_naive1, mse_naive2, mse_naive3]

rmse_spike = [rmse_mean_spike, rmse_naive1_spike, rmse_naive2_spike, rmse_naive3_spike]
mae_spike = [mae_mean_spike, mae_naive1_spike, mae_naive2_spike, mae_naive3_spike]
mse_spike = [mse_mean_spike, mse_naive1_spike, mse_naive2_spike, mse_naive3_spike]

rmse_normal = [rmse_mean_normal, rmse_naive1_normal, rmse_naive2_normal, rmse_naive3_normal]
mae_normal = [mae_mean_normal, mae_naive1_normal, mae_naive2_normal, mae_naive3_normal]
mse_normal = [mse_mean_normal, mse_naive1_normal, mse_naive2_normal, mse_naive3_normal]


results = pd.DataFrame({'rmse': rmse,
                        'mae': mae, 
                        'rmse_spike': rmse_spike, 
                        'rmse_normal': rmse_normal,
                        'mae_spike': mae_spike,
                        'mae_normal': mae_normal})

results['index'] = ['Mean_benchmark', 'Naive_1', 'Naive_2', 'Naive_3']
results.set_index('index', inplace = True)

results.to_csv('Results_Benchmarks.csv')

# =============================================================================
# Plotting results
# =============================================================================

# plot window
w_plot = 96
fontsize = 13

Residual = list(y_test) - y_pred_mean

plt.figure(figsize=(11,6))
plt.subplot(2, 2, 1)
plt.plot(np.arange(0, (w_plot)), y_test[-w_plot:], label = 'Real values', linewidth = 1.5, color = 'steelblue')
plt.plot(np.arange(0, (w_plot)), y_pred_mean[-w_plot:], label = 'Predicted values', linewidth = 1.2, color= 'deepskyblue')
plt.plot(np.arange(0, (w_plot)), Residual[-w_plot:], label = 'Residual error', linewidth = 0.8, color = 'slategrey')
plt.fill_between(np.arange(0, (w_plot)),  data['spike_lowerlim'][-w_plot:],data['spike_upperlim'][-w_plot:], facecolor='skyblue', alpha=0.5, label = 'Not spike regions')
plt.xlim(0, w_plot - 1)
plt.ylim(-160, 260)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.title('Benchmark 1: Training Mean', fontsize = fontsize + 2)
plt.tight_layout()

Residual = list(y_test) - y_naive1

plt.subplot(2, 2, 2)
plt.plot(np.arange(0, (w_plot)), y_test[-w_plot:], label = 'Real values', linewidth = 1.5, color = 'steelblue')
plt.plot(np.arange(0, (w_plot)), y_naive1[-w_plot:], label = 'Predicted values', linewidth = 1.2, color= 'deepskyblue')
plt.plot(np.arange(0, (w_plot)), Residual[-w_plot:], label = 'Residual error', linewidth = 0.8, color = 'slategrey')
plt.fill_between(np.arange(0, (w_plot)),  data['spike_lowerlim'][-w_plot:],data['spike_upperlim'][-w_plot:], facecolor='skyblue', alpha=0.5, label = 'Not spike regions')
plt.xlim(0, w_plot - 1)
plt.ylim(-160, 260)
plt.xticks(fontsize = 20)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.title('Benchmark 2: Previous SP', fontsize = fontsize + 2)
plt.tight_layout()

Residual = list(y_test) - y_naive2

plt.subplot(2, 2, 3)
plt.plot(np.arange(0, (w_plot)), y_test[-w_plot:], label = 'Real values', linewidth = 1.5, color = 'steelblue')
plt.plot(np.arange(0, (w_plot)), y_naive2[-w_plot:], label = 'Predicted values', linewidth = 1.2, color= 'deepskyblue')
plt.plot(np.arange(0, (w_plot)), Residual[-w_plot:], label = 'Residual error', linewidth = 0.8, color = 'slategrey')
plt.fill_between(np.arange(0, (w_plot)),  data['spike_lowerlim'][-w_plot:],data['spike_upperlim'][-w_plot:], facecolor='skyblue', alpha=0.5, label = 'Not spike regions')
plt.xlim(0, w_plot - 1)
plt.ylim(-160, 260)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.title('Benchmark 3: Previous day SP', fontsize = fontsize + 2)
plt.tight_layout()

Residual = list(y_test) - y_naive3

plt.subplot(2, 2, 4)
plt.plot(np.arange(0, (w_plot)), y_test[-w_plot:], label = 'Real values', linewidth = 1.5, color = 'steelblue')
plt.plot(np.arange(0, (w_plot)), y_naive3[-w_plot:], label = 'Predicted values', linewidth = 1.2, color= 'deepskyblue')
plt.plot(np.arange(0, (w_plot)), Residual[-w_plot:], label = 'Residual error', linewidth = 0.8, color = 'slategrey')
plt.fill_between(np.arange(0, (w_plot)),  data['spike_lowerlim'][-w_plot:],data['spike_upperlim'][-w_plot:], facecolor='skyblue', alpha=0.5, label = 'Not spike regions')
plt.xlim(0, w_plot -1)
plt.ylim(-160, 260)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.title('Benchmark 4: Previous week SP', fontsize = fontsize + 2)
plt.tight_layout()
plt.savefig('Plot_all_4_Benchmarks.png')

# save legend
plt.figure(figsize=(10,4))
plt.plot(np.arange(0, (w_plot)), y_test[-w_plot:], label = 'Real values', linewidth = 1.5, color = 'steelblue')
plt.plot(np.arange(0, (w_plot)), y_naive3[-w_plot:], label = 'Predicted values', linewidth = 1.2, color= 'deepskyblue')
plt.plot(np.arange(0, (w_plot)), Residual[-w_plot:], label = 'Residual error', linewidth = 0.8, color = 'slategrey')
plt.fill_between(np.arange(0, (w_plot)),  data['spike_lowerlim'][-w_plot:],data['spike_upperlim'][-w_plot:], facecolor='skyblue', alpha=0.5, label = 'Not spike regions')
plt.xlim(0, w_plot -1)
plt.ylim(-160, 260)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Accumulated SP', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.title('Benchmark 4: Previous week SP', fontsize = fontsize + 2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.savefig('Plot_Benchmark_Legend.png')

