# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:15:36 2020

@author: maria
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cols1 = ['HDR', 'market_index_price']

for i in range(1, 13):    
    data4 = pd.read_csv('Data/MarketIndexData_20200525_1812({}).csv'.format(i))
    data4.reset_index( inplace = True)
    data4.drop('level_0', axis = 'columns', inplace = True)
    data4.drop('level_1', axis = 'columns', inplace = True)
    data4.dropna(axis = 0, inplace = True)
    data4['level_2'] = data4['level_2'].astype(int)
    data4['level_3'] = data4['level_3'].astype(int)
    data4['level_2'] = data4['level_2'].astype(str) + data4['level_3'].astype(str)
    data4.drop('level_3', axis = 1, inplace = True)
    data4.set_index('level_2', inplace = True)
    data4.index.name = 'YYYY/MM/DD/SP'
    data4.columns = ['price', 'volume_mwh']
    data4 = data4[(data4.T != 0).any()]
    if i == 1:
        append_here = data4
    else:
        append_here = pd.concat([data4, append_here], axis = 0, join = 'inner')

mid_final = append_here
mid_prediction = mid_final.shift(periods = 1, axis = 0, fill_value = np.nan)
delta = mid_final - mid_prediction

fig = plt.figure()

fig.subplots_adjust(bottom = 0.00, left = 0.10, top = 1.500, right = 1.50)
plt.subplot(2, 2, 1)
#plt.title('Naive method applied to predict both Market index price and Volume of electricity')
mid_final['price'].plot(label = 'Real price', color = 'black')
mid_prediction['price'].plot(label = 'Predicted price', color = 'lightblue')
plt.legend()
plt.xticks(np.linspace(0, 614, 12), list(range(14, 26)))
plt.xlabel('Days of May 2020')
plt.ylabel('Real Market Index Price in £/Mwh')

plt.subplot(2, 2, 2)
mid_final['volume_mwh'].plot(label = 'Real Volume', color = 'black')
mid_prediction['volume_mwh'].plot(label = 'Predicted volume', color = 'lightblue')
plt.xticks(np.linspace(0, 614, 12), list(range(14, 26)))
plt.legend()
plt.xlabel('Days of May 2020')
plt.ylabel('Volume in Mwh')

plt.subplot(2, 2, 3)
delta['price'].plot(label = 'Delta = Real - prediction', color = 'orange')
plt.xticks(np.linspace(0, 614, 12), list(range(14, 26)))
plt.legend()
plt.xlabel('Days of May 2020')
plt.subplot(2, 2, 4)
delta['volume_mwh'].plot(label = 'Delta = Real - prediction', color = 'orange')
plt.legend()
plt.xlabel('Days of May 2020')
plt.xticks(np.linspace(0, 614, 12), list(range(14, 26)))

