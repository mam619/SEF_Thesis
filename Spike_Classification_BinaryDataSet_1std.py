# =============================================================================
# Spike occurences definition (Binary data set) for different windows
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values

data.fillna(method = 'ffill', inplace = True)

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
from pylab import rcParams

results = seasonal_decompose(offers18, model = 'aditive', period = 336)
# Add the same figure size to all plots
# rcParams['figure.figsize'] = 12,5
# results.plot()

# No trend and no season observed
'''
# =============================================================================
# Spike binary label w/ mean +- 1 std filter
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
    # create the moving average data set with correspondent rolling window
    data18['sma'] = data18['Offers'].rolling(window = w).mean()
    # create the standard deviation data set with correspondent rolling window
    data18['std'] = data18['Offers'].rolling(w).std()
    # create upper and lower limits data set
    data18['spike_upperlim'] = data18['sma'] + (data18['std'])
    data18['spike_lowerlim'] = data18['sma'] - (data18['std'])
    # create binary data set with occurences out of these limis
    data18['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)
    
    # for each rolling window count the number of spikes and normal occurences
    num_spikes_18.append((data18['spike_occurance'] == 1).sum())
    num_normal_18.append((data18['spike_occurance'] == 0).sum())
    
    # same process for the year of 2017
    data17['sma'] = data17['Offers'].rolling(window = w).mean()
    data17['std'] = data17['Offers'].rolling(w).std()
    data17['spike_upperlim'] = data17['sma'] + (data17['std'])
    data17['spike_lowerlim'] = data17['sma'] - (data17['std'])
    data17['spike_occurance'] = ((data17['Offers'] > data17['spike_upperlim']) | (data17['Offers'] < data17['spike_lowerlim'])).astype(np.int)
    
    num_spikes_17.append((data17['spike_occurance'] == 1).sum())
    num_normal_17.append((data17['spike_occurance'] == 0).sum())
    
    # same process for the year of 2016
    data16['sma'] = data16['Offers'].rolling(window = w).mean()
    data16['std'] = data16['Offers'].rolling(w).std()
    data16['spike_upperlim'] = data16['sma'] + (data16['std'])
    data16['spike_lowerlim'] = data16['sma'] - (data16['std'])
    data16['spike_occurance'] = ((data16['Offers'] > data16['spike_upperlim']) | (data16['Offers'] < data16['spike_lowerlim'])).astype(np.int)
    
    num_spikes_16.append((data16['spike_occurance'] == 1).sum())
    num_normal_16.append((data16['spike_occurance'] == 0).sum())
    window.append(w)

# create DataSet for each year with number of spikes for diferent rolling windows  
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
plt.ylabel('Number of SP with spikes')
plt.xlabel('Window number (SP)')
plt.xticks(np.arange(0, 62, 2))
plt.scatter(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window, max_spike_18 , color = 'red', label = 'Maximum values')
plt.text(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window , max_spike_18 + 200, '({},{})'.format(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window.iloc[0], max_spike_18))
plt.scatter(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window, max_spike_17 , color = 'red')
plt.text(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window + 1, max_spike_17 - 200, '({},{})'.format(spike_var_17[spike_var_17['num_spikes'] == max_spike_17].window.iloc[0], max_spike_17))
plt.scatter(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window, max_spike_16 , color = 'red')
plt.text(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window , max_spike_16 + 200, '({},{})'.format(spike_var_16[spike_var_16['num_spikes'] == max_spike_16].window.iloc[0], max_spike_16))
plt.title('Number of SP with spike occurances according\n to recursive filter (RF) for different \nrolling windows for three different years\n', fontsize= 15)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.legend()
plt.show()

# plot of both spike occurences and normal opperation SP together
# 2018
plt.plot(range_w, spike_var_18['num_spikes'], label = 'Number of SP with spike occurences')
plt.plot(range_w, spike_var_18['num_normal'], label = 'Number of SP under normal operation')
plt.ylabel('Number of SP')
plt.xlabel('Window number (SP)')
plt.title('Number of SP with spike occurances and under normal operation for the year of 2018 \n plotted for different rolling windows according \n to recursive filter (RF) for  different rolling windows\n', fontsize= 15)
plt.xticks(np.arange(0, 62, 2))
#plt.scatter(spike_var[spike_var['num_normal'] == max_num].window, max_num , color = 'red')
plt.scatter(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window, max_spike_18 , color = 'red', label = 'Maximum value')
plt.text(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window + 0.5, max_spike_18 + 0.5, '({},{})'.format(spike_var_18[spike_var_18['num_spikes'] == max_spike_18].window.iloc[0], max_spike_18))
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.legend()
plt.show()

# =============================================================================
# 
# # =============================================================================
# # Some visualisation
# # =============================================================================
# 
# w = 48
# offers1850 = data18
# plt.figure(figsize=(12,4))
# offers1850['sma'] = data18['Offers'].rolling(window = w).mean()
# offers1850['std'] = data18['Offers'].rolling(w).std()
# offers1850['spike_upperlim'] = data18['sma'] + (data18['std'])
# offers1850['spike_lowerlim'] = data18['sma'] - (data18['std'])
# offers1850['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)
# 
# plt.plot(data18[-49:]['Offers'], label = 'Offer')
# plt.plot(data18[-49:]['sma'], label = 'Simple moving average')
# plt.plot(data18[-49:]['spike_upperlim'], label = ' Spike upper limit')
# plt.plot(data18[-49:]['spike_lowerlim'], label = ' Spike lower limit')
# plt.title('Offers with spike limits for the last day of 2018 with a rolling window = {} SP'.format(w))
# plt.xticks(np.arange(17473, 17524, 1), np.arange(0, 49), fontsize = 10)
# plt.legend(fontsize = 14)
# plt.minorticks_on()
# plt.grid(which='major', linestyle='-', linewidth='0.5')
# plt.grid(which='minor', linestyle=':', linewidth='0.5')
# plt.ylim(0,220)
# plt.xlim(17472,17522)
# plt.ylabel('Offer price in £/MWh', fontsize = 14)
# plt.xlabel('SP of the day', fontsize = 14)
# plt.tight_layout()
# plt.show()
# 
# w = 8
# plt.figure(figsize=(12,4))
# data18['sma'] = data18['Offers'].rolling(window = w).mean()
# data18['std'] = data18['Offers'].rolling(w).std()
# data18['spike_upperlim'] = data18['sma'] + (data18['std'])
# data18['spike_lowerlim'] = data18['sma'] - (data18['std'])
# data18['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)
# plt.plot(data18[-49:]['Offers'], label = 'Offer')
# plt.plot(data18[-49:]['sma'], label = 'Simple moving average')
# plt.plot(data18[-49:]['spike_upperlim'], label = ' Spike upper limit')
# plt.plot(data18[-49:]['spike_lowerlim'], label = ' Spike lower limit')
# plt.title('Offers with spike limits for the last day of 2018 with a rolling window = {} SP'.format(w))
# plt.xticks(np.arange(17473, 17524, 1), np.arange(0, 49), fontsize = 10)
# plt.legend(fontsize = 14)
# plt.minorticks_on()
# plt.grid(which='major', linestyle='-', linewidth='0.5')
# plt.grid(which='minor', linestyle=':', linewidth='0.5')
# plt.ylim(0,220)
# plt.xlim(17472,17522)
# plt.ylabel('Offer price in £/MWh', fontsize = 14)
# plt.xlabel('SP of the day', fontsize = 14)
# plt.tight_layout()
# plt.show()
# 
# 
# # another day 
# 
# w = 48
# offers1850 = data18
# plt.figure(figsize=(15,5))
# offers1850['sma'] = data18['Offers'].rolling(window = w).mean()
# offers1850['std'] = data18['Offers'].rolling(w).std()
# offers1850['spike_upperlim'] = data18['sma'] + (data18['std'])
# offers1850['spike_lowerlim'] = data18['sma'] - (data18['std'])
# offers1850['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)
# 
# plt.plot(data18[-144:-95]['Offers'], label = 'Offer')
# plt.plot(data18[-144:-95]['sma'], label = 'Simple moving average')
# plt.plot(data18[-144:-95]['spike_upperlim'], label = ' Spike upper limit')
# plt.plot(data18[-144:-95]['spike_lowerlim'], label = ' Spike lower limit')
# plt.title('Offers with spike limits for the last day of 2018 with a rolling window = {} SP'.format(w))
# plt.legend()
# plt.minorticks_on()
# plt.grid(which='major', linestyle='-', linewidth='0.5')
# plt.grid(which='minor', linestyle=':', linewidth='0.5')
# plt.xticks(np.arange(17378, 17427, 1), np.arange(0, 49))
# plt.xlim(17377,17428)
# plt.ylim(0,290)
# plt.ylabel('Offer price in £/MWh')
# plt.xlabel('SP of the day')
# plt.show()
# =============================================================================
'''

# =============================================================================
# Save spike bin data set with rolling window = 48
# =============================================================================

w = 48
plt.figure(figsize=(15,5))
data['sma'] = data['Offers'].rolling(window = w).mean()
data['std'] = data['Offers'].rolling(w).std()
data['spike_upperlim'] = data['sma'] + (data['std'])
data['spike_lowerlim'] = data['sma'] - (data['std'])
data['spike_occurance'] = ((data['Offers'] > data['spike_upperlim']) | (data['Offers'] < data['spike_lowerlim'])).astype(np.int)

'''
data_to_save = data.iloc[:,-6:]

data_to_save.to_csv('Spike_binary_1std.csv')
'''

# divide data by years again
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

n = int(30.5 * 48) 

split_month = lambda datayear, n = n : [datayear[i: i + n] for i in range(0, len(datayear), n)]

spike_occurance_18 = split_month(data18.spike_occurance)
spike_occurance_17 = split_month(data17.spike_occurance)
spike_occurance_16 = split_month(data16.spike_occurance)

count_18 = []
perc_18 = []
count_17 = []
perc_17 = []
count_16 = []
perc_16 = []

for i in range(0, len(spike_occurance_18)):
    count_18.append((spike_occurance_18[i] == 1).sum())
    perc_18.append(100 * count_18[-1] / len(spike_occurance_18[i]))
    count_17.append((spike_occurance_17[i] == 1).sum())
    perc_17.append(100 * count_17[-1] / len(spike_occurance_17[i]))
    count_16.append((spike_occurance_16[i] == 1).sum())
    perc_16.append(100 * count_16[-1] / len(spike_occurance_16[i]))
    

# PLOT RESULTS
fig, ax1 = plt.subplots(figsize=(11,4))

ax1.set_xlabel('Months of the year')
ax1.set_ylabel('Number of SP')
ax1.minorticks_on()
ax1.grid(which='major', linestyle='-', linewidth='0.5')
ax1.grid(which='minor', linestyle=':', linewidth='0.5')
ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], count_18, color = 'steelblue', label = '2018')
ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], count_17, color = 'darkorange', label = '2017')
ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], count_16, color = 'green', label = '2016')
ax1.set_xticks([1,2,3,4,5,6,7,8,9, 10, 11,12])
ax1.set_xlim([0.5,12.5])
ax1.legend()
ax1.set_title('Number of spikes per month for 2018, 2017 and 2019 and respective percentages')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Percentage of SP (%)')
ax2.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], perc_18, color = 'none')
ax2.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], perc_17, color = 'none')
ax2.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], perc_16, color = 'none')

plt.rcParams.update({'font.size': 13})

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
fig.savefig('Spikes_per_month.png')

print("Total number of spikes for 2018 were {}".format(sum(count_18)))
print("Total number of spikes for 2017 were {}".format(sum(count_17)))
print("Total number of spikes for 2016 were {}".format(sum(count_16)))

