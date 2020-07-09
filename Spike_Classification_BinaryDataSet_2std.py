# =============================================================================
# Spike occurences definition (Binary data set) for different windows
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = pd.read_csv('Feature_Handeling/Features_ARENKO.csv', index_col = 0)
features_2 = pd.read_csv('Feature_Handeling/Features_APIs.csv', index_col = 0)
offers = pd.read_csv('Feature_Handeling/UK__Offers.csv', index_col = 0)
# bids = pd.read_csv('WORKSHOP_Ioannis/Bids.csv', index_col = 0)

# combine both offers and features together
data = pd.concat([features, features_2, offers], axis=1, sort=True)

# shift offers two SP back for realistic predictions
# data['Offers'] = data['Offers'].shift(-2) NOT SURE IS CORRECT

# filter any offer higher than 6000 out
offers = offers[offers < 6000]

# fill missing values
data.fillna(value = data.mean(), inplace = True)

# collect data for each year
data18 = data.loc[data.index > 2018000000, :]
data17 = data.loc[(data.index < 2018000000) & (data.index > 2017000000), :]
data16 = data.loc[data.index < 2017000000, :]

# reset the index so it does not influence the plots
data18 = data18.reset_index()
data17 = data17.reset_index()
data16 = data16.reset_index()

# collect offers from each year's data set
offers18 = data18['Offers']
offers17 = data17['Offers']
offers16 = data16['Offers']

# =============================================================================
# ETS decomposition
# =============================================================================
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(offers18, model = 'aditive', period = 336)
from pylab import rcParams
# Add the same figure size to all plots
# rcParams['figure.figsize'] = 12,5
#results.plot()
# No trend and no season observed

# =============================================================================
# Spike binary label w/ mean +- 2 std filter
# =============================================================================

# set up 
range_w = np.arange(2, 60, 1)
window = []
num_spikes_18 = []
num_normal_18 = []
num_spikes_17 = []
num_normal_17 = []
num_spikes_16 = []
num_normal_16 = []

# create DataSet to understand number of spikes for different rolling windows
for w in range_w:
    # 2018
    data18['sma'] = data18['Offers'].rolling(window = w).mean()
    data18['std'] = data18['Offers'].rolling(w).std()
    data18['spike_upperlim'] = data18['sma'] + (2 * data18['std'])
    data18['spike_lowerlim'] = data18['sma'] - (2 * data18['std'])
    data18['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)
    
    num_spikes_18.append((data18['spike_occurance'] == 1).sum())
    num_normal_18.append((data18['spike_occurance'] == 0).sum())
    
    # 2017
    data17['sma'] = data17['Offers'].rolling(window = w).mean()
    data17['std'] = data17['Offers'].rolling(w).std()
    data17['spike_upperlim'] = data17['sma'] + (2 * data17['std'])
    data17['spike_lowerlim'] = data17['sma'] - (2 * data17['std'])
    data17['spike_occurance'] = ((data17['Offers'] > data17['spike_upperlim']) | (data17['Offers'] < data17['spike_lowerlim'])).astype(np.int)
    
    num_spikes_17.append((data17['spike_occurance'] == 1).sum())
    num_normal_17.append((data17['spike_occurance'] == 0).sum())
    
    # 2016
    data16['sma'] = data16['Offers'].rolling(window = w).mean()
    data16['std'] = data16['Offers'].rolling(w).std()
    data16['spike_upperlim'] = data16['sma'] + (2 * data16['std'])
    data16['spike_lowerlim'] = data16['sma'] - (2 * data16['std'])
    data16['spike_occurance'] = ((data16['Offers'] > data16['spike_upperlim']) | (data16['Offers'] < data16['spike_lowerlim'])).astype(np.int)
    
    num_spikes_16.append((data16['spike_occurance'] == 1).sum())
    num_normal_16.append((data16['spike_occurance'] == 0).sum())
    window.append(w)

# create DataSet with number of spikes for diferent rolling windows  
spike_var_18 = pd.DataFrame({'window': window, 'num_spikes': num_spikes_18, 'num_normal' : num_normal_18})
spike_var_18['total'] = spike_var_18['num_spikes'] + spike_var_18['num_normal']

spike_var_17 = pd.DataFrame({'window': window, 'num_spikes': num_spikes_17, 'num_normal' : num_normal_17})
spike_var_17['total'] = spike_var_17['num_spikes'] + spike_var_17['num_normal']

spike_var_16 = pd.DataFrame({'window': window, 'num_spikes': num_spikes_16, 'num_normal' : num_normal_16})
spike_var_16['total'] = spike_var_16['num_spikes'] + spike_var_16['num_normal']

# =============================================================================
# Plot number of spikes and number of normal occurances vs rolling window 
# =============================================================================

# calculating maximums to after plot for each year
max_num_18 = spike_var_18['num_normal'].max()
max_spike_18 = spike_var_18['num_spikes'].max()
max_num_17 = spike_var_17['num_normal'].max()
max_spike_17 = spike_var_17['num_spikes'].max()
max_num_16 = spike_var_16['num_normal'].max()
max_spike_16 = spike_var_16['num_spikes'].max()

# spike occurences plot

# 2018, 2017, 2016
rcParams['figure.figsize'] = 15,5
plt.plot(range_w, spike_var_18['num_spikes'], label = '2018')
plt.plot(range_w, spike_var_17['num_spikes'], label = '2017')
plt.plot(range_w, spike_var_16['num_spikes'], label = '2016')
plt.ylabel('Number of SP')
plt.xlabel('Window number')
plt.xticks(np.arange(0, 62, 2))
plt.scatter(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window, max_spike_18 , color = 'red', label = 'Maximum values')
plt.text(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window + 0.5, max_spike_18 + 0.5, '({},{})'.format(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window.iloc[0], max_spike_18))
plt.scatter(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window, max_spike_17 , color = 'red')
plt.text(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window + 0.5, max_spike_17 + 0.5, '({},{})'.format(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window.iloc[0], max_spike_17))
plt.scatter(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window, max_spike_16 , color = 'red')
plt.text(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window + 0.5, max_spike_16 + 0.5, '({},{})'.format(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window.iloc[0], max_spike_16))
plt.title('Number of SP with spike occurances according\n to recursive filter (RF) for different \nrolling windows for three different years\n', fontsize= 15)
plt.minorticks_on() #required for the minor grid
plt.grid(which = 'major', linestyle ='-', linewidth = '0.25', color = 'black')
#plt.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
plt.legend()
plt.show()

'''
# 2017
rcParams['figure.figsize'] = 10,5
plt.plot(range_w, spike_var_17['num_spikes'])
plt.ylabel('Number of SP')
plt.xlabel('Window number')
plt.xticks(np.arange(0, 62, 2))
plt.scatter(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window, max_spike_17 , color = 'red', label = 'Maximum value')
plt.text(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window[0] + 0.5, max_spike_17 + 0.5, '({},{})'.format(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window.iloc[0], max_spike_17))
plt.title('Number of SP with spike occurances according\n to recursive filter (RF) for different rolling windows on 2017\n', fontsize= 15)
plt.grid(which = 'major', linestyle ='-', linewidth = '0.25', color = 'black')
plt.legend()
plt.show()

# 2016
rcParams['figure.figsize'] = 10,5
plt.plot(range_w, spike_var_16['num_spikes'])
plt.ylabel('Number of SP')
plt.xlabel('Window number')
plt.xticks(np.arange(0, 62, 2))
plt.scatter(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window, max_spike_16 , color = 'red', label = 'Maximum value')
plt.text(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window + 0.5, max_spike_16 + 0.5, '({},{})'.format(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window.iloc[0], max_spike_16))
plt.title('Number of SP with spike occurances according\n to recursive filter (RF) for different rolling windows on 2016\n', fontsize= 15)
plt.grid(which = 'major', linestyle ='-', linewidth = '0.25', color = 'black')
plt.legend()
plt.show()
'''

# plot of both spike occurences and normal opperation SP together
# 2018
plt.plot(range_w, spike_var_18['num_spikes'], label = 'Number of SP with spike occurences')
plt.plot(range_w, spike_var_18['num_normal'], label = 'Number of SP under normal operation')
plt.ylabel('Number of SP')
plt.xlabel('Window number')
plt.title('Number of SP with spike occurances and under normal operation for the year of 2018 \n plotted for different rolling windows according \n to recursive filter (RF) for  different rolling windows\n', fontsize= 15)
plt.xticks(np.arange(0, 62, 2))
#plt.scatter(spike_var[spike_var['num_normal'] == max_num].window, max_num , color = 'red')
plt.scatter(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window, max_spike_18 , color = 'red', label = 'Maximum value')
plt.text(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window + 0.5, max_spike_18 + 0.5, '({},{})'.format(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window.iloc[0], max_spike_18))
plt.grid(which = 'major', linestyle ='-', linewidth = '0.25', color = 'black')
plt.legend()
plt.show()

# =============================================================================
# Save binary data set
# =============================================================================
w = 48
plt.figure(figsize=(15,5))
data['sma'] = data['Offers'].rolling(window = w).mean()
data['std'] = data['Offers'].rolling(w).std()
data['spike_upperlim'] = data['sma'] + (2 * data['std'])
data['spike_lowerlim'] = data['sma'] - (2 * data['std'])
data['spike_occurance'] = ((data['Offers'] > data['spike_upperlim']) | (data['Offers'] < data['spike_lowerlim'])).astype(np.int)

data_to_save = data.iloc[:,-6:]

data_to_save.to_csv('Spike_binary_2std.csv')

# =============================================================================
# Visualisation of data for different time gaps
# =============================================================================
w = 50
offers1850 = data18
plt.figure(figsize=(15,5))
offers1850['sma'] = data18['Offers'].rolling(window = w).mean()
offers1850['std'] = data18['Offers'].rolling(w).std()
offers1850['spike_upperlim'] = data18['sma'] + (2 * data18['std'])
offers1850['spike_lowerlim'] = data18['sma'] - (2 * data18['std'])
offers1850['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)

plt.plot(offers1850[-48:]['Offers'], label = 'Offer')
plt.plot(offers1850[-48:]['sma'], label = 'Simple moving average')
plt.plot(offers1850[-48:]['spike_upperlim'], label = ' Spike upper limit')
plt.plot(offers1850[-48:]['spike_lowerlim'], label = ' Spike lower limit')
plt.title('Offers with spike limits for the last day of 2018 with a rolling window = {} SP'.format(w))
plt.xticks(np.arange(17474, 17523, 2), np.arange(0, 26))
plt.legend()
plt.ylim(0,220)
plt.ylabel('Offer price in £/MWh')
plt.xlabel('Hours of the day')
plt.show()

w = 8
plt.figure(figsize=(15,5))
data18['sma'] = data18['Offers'].rolling(window = w).mean()
data18['std'] = data18['Offers'].rolling(w).std()
data18['spike_upperlim'] = data18['sma'] + (2 * data18['std'])
data18['spike_lowerlim'] = data18['sma'] - (2 * data18['std'])
data18['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)
plt.plot(data18[-48:]['Offers'], label = 'Offer')
plt.plot(data18[-48:]['sma'], label = 'Simple moving average')
plt.plot(data18[-48:]['spike_upperlim'], label = ' Spike upper limit')
plt.plot(data18[-48:]['spike_lowerlim'], label = ' Spike lower limit')
plt.title('Offers with spike limits for the last day of 2018 with a rolling window = {} SP'.format(w))
plt.xticks(np.arange(17474, 17523, 2), np.arange(0, 26))
plt.legend()
plt.ylim(0,220)
plt.ylabel('Offer price in £/MWh')
plt.xlabel('Hours of the day')
plt.show()