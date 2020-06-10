# =============================================================================
# Exercise on spike prediction
# =============================================================================

# import all libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# download data sets
features = pd.read_csv('WORKSHOP_Ioannis/System Features.csv', index_col = 0)
offers = pd.read_csv('WORKSHOP_Ioannis/Offers.csv', index_col = 0)

# combine both offers and features together
#bids = pd.read_csv('WORKSHOP_Ioannis/Bids.csv', index_col = 0)
data = pd.concat([features, offers], axis=1, sort=True)

# shift offers two SP back for realistic predictions
data['Offers'] = data['Offers'].shift(-2)

# filter any offer higher than 6000 out
offers = offers[offers < 6000]

# fill missing values and collect data for 2018 only
data.fillna(value = data.mean(), inplace = True)
data18 = data.loc[data.index > 2018000000, :]

# reset the index so it does not influence the plots
data18 = data18.reset_index()

# collect offers from 2018
offers18 = data18['Offers']

# to check if there is Nan values
# offers18.isna().sum()
# bids18.isna().sum()

# =============================================================================
# ETS decomposition
# =============================================================================
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(offers18, model = 'aditive', period = 336)
from pylab import rcParams
# Add the same figure size to all plots
rcParams['figure.figsize'] = 12,5
#results.plot()
# No trend and no season observed

# =============================================================================
# Spike binary label w/ mean +- 2 std filter
# =============================================================================

# set up 
range_w = np.arange(2, 60, 1)
window = []
num_spikes = []
num_normal = []
total = []

# create DataSet to understand number of spikes for different rolling windows
for w in range_w:
    data18['sma'] = data18['Offers'].rolling(window = w).mean()
    data18['std'] = data18['Offers'].rolling(w).std()
    data18['spike_upperlim'] = data18['sma'] + (2 * data18['std'])
    data18['spike_lowerlim'] = data18['sma'] - (2 * data18['std'])
    data18['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)
    window.append(w)
    num_spikes.append((data18['spike_occurance'] == 1).sum())
    num_normal.append((data18['spike_occurance'] == 0).sum())
    total.append
spike_var = pd.DataFrame({'window': window, 'num_spikes': num_spikes, 'num_normal' : num_normal})
spike_var['total'] = spike_var['num_spikes'] + spike_var['num_normal']

# plot number of spikes and number of normal occurances vs rolling window 

# calculating maximums to after plot
max_num = spike_var['num_normal'].max()
max_spike = spike_var['num_spikes'].max()

# spike occurences plot
rcParams['figure.figsize'] = 10,5
plt.plot(range_w, spike_var['num_spikes'])
plt.ylabel('Number of SP')
plt.xlabel('Window number')
plt.xticks(np.arange(2, 60, 2))
plt.legend()
plt.scatter(spike_var[spike_var['num_spikes'] == max_spike].window, max_spike , color = 'red')
plt.title('Number of SP with spike occurances according\n to recursive filter (RF) for different rolling windows\n', fontsize= 15)
plt.show()

'''
# normal occurences plot
plt.plot(range_w, spike_var['num_normal'])
plt.ylabel('Number of SP')
plt.xlabel('Window number')
plt.legend()
#plt.scatter(spike_var[spike_var['num_normal'] == max_num].window, max_num , color = 'red')
plt.title('Number of SP under normal operation according \n to recursive filter (RF) for  different rolling windows\n', fontsize= 15)
plt.xticks(np.arange(2, 60, 2))
plt.show()
'''

# both together
plt.plot(range_w, spike_var['num_spikes'], label = 'Number of SP with spike occurences')
plt.plot(range_w, spike_var['num_normal'], label = 'Number of SP under normal operation')
plt.ylabel('Number of SP')
plt.xlabel('Window number')
plt.title('Number of SP with spike occurances and under normal operation for the year of 2018 \n plotted for different rolling windows according \n to recursive filter (RF) for  different rolling windows\n', fontsize= 15)
plt.xticks(np.arange(2, 60, 2))
plt.legend()
#plt.scatter(spike_var[spike_var['num_normal'] == max_num].window, max_num , color = 'red')
plt.scatter(spike_var[spike_var['num_spikes'] == max_spike].window, max_spike , color = 'red')
plt.show()

# create final data set with desired w
w = 50
offers1850 = data18
plt.figure(figsize=(15,5))
offers1850['sma'] = data18['Offers'].rolling(window = w).mean()
offers1850['std'] = data18['Offers'].rolling(w).std()
offers1850['spike_upperlim'] = data18['sma'] + (2 * data18['std'])
offers1850['spike_lowerlim'] = data18['sma'] - (2 * data18['std'])
offers1850['spike_occurance'] = ((data18['Offers'] > data18['spike_upperlim']) | (data18['Offers'] < data18['spike_lowerlim'])).astype(np.int)
plt.plot(offers1850[-48:]['Offers'])
plt.plot(offers1850[-48:]['sma'], label = 'Simple moving average')
plt.plot(offers1850[-48:]['spike_upperlim'], label = ' Spike upper limit')
plt.plot(offers1850[-48:]['spike_lowerlim'], label = ' Spike lower limit')
plt.title('Offers with spike limits for the last day of 2018 with a rolling window = {}'.format(w))
plt.xticks(np.arange(17474, 17521, 2), np.arange(0, 25))
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
plt.plot(data18[-48:]['Offers'])
plt.plot(data18[-48:]['sma'], label = 'Simple moving average')
plt.plot(data18[-48:]['spike_upperlim'], label = ' Spike upper limit')
plt.plot(data18[-48:]['spike_lowerlim'], label = ' Spike lower limit')
plt.title('Offers with spike limits for the last day of 2018 with a rolling window = {}'.format(w))
plt.xticks(np.arange(17474, 17521, 2), np.arange(0, 25))
plt.legend()
plt.ylim(0,220)
plt.ylabel('Offer price in £/MWh')
plt.xlabel('Hours of the day')
plt.show()

# =============================================================================
# Some initial classification predictions
# =============================================================================

# set X and y
X = data18.iloc[:, 1:-6]
y = data18['spike_occurance']

# divide data into train and test with 10% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.4, shuffle=False)

# feature scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# use KNN method for different k values
k_range = range(1, 26)
scores = []

from sklearn.metrics import accuracy_score

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = pd.Series(knn.predict(X_test))
    scores.append(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

# plot results
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

# =============================================================================
# cunfusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

(y_test == 1).sum() # 549
(y_pred == 1).sum() # 15
# model only has 2 TN (spikes well predicted)

spike_accuracy = cm[1][1]/(y_test == 1).sum() # 0.0013 for window = 6
spike_confidence = cm[1][1]/(y_pred == 1).sum() # 0.13 for window = 6
print('Spike accuracy is {}'.format(spike_accuracy))
print('Spike confidence is {}'.format(spike_confidence))

