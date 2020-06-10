# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:57:09 2020

Downloading and rearanging trial data from ELEXON

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Imbalance prices csv for 01 - 03 - 2020

col1 = ['Time Series ID', 'Business Type', 'Set Date', 'SP', 'Imb Price']
data1 = pd.read_csv('Imbalance_Prices_20200301.csv', names = col1,
                    usecols = col1[2:5])

data1.columns = data1.columns.str.replace(' ', '_')
data1.columns = data1.columns.str.lower()
data1.drop([0,1], axis = 0, inplace = True)
data1 = data1.loc[data1.sp.duplicated(keep ='first'), :]
data1 = data1.reset_index(drop = True)
# data1.drop_duplicates(keeps = 'last').shape
# data1['imbalance_price_amount(gbp)'].describe()
data1['imb_price'] = data1.imb_price.astype(float)
#data1.imb_price.value_counts().sort_values()
#data1.imb_price.plot(kind = 'hist')
#data1.imb_price.value_counts().sort_values().plot(kind = 'bar')

data1.set_date = data1.set_date.str.replace('-', '') + data1.sp
data1.set_index('set_date', inplace = True)
data1.index.name = 'YYYY/MM/DD/SP'
data1.drop('sp', axis = 1, inplace = True)

plt.plot(data1)
plt.ylabel('£/MW')
plt.xticks(list(range(0,49, 4)), list(range(0, 26, 2)))
plt.xlim(0, 48)
plt.title('Imbalance price on 01 - 03 - 2020')
plt.xlabel('Hours of the day')
plt.show()

# =============================================================================

# Generation by Fuel type for 01 - 03 - 2020

col2 = ['x','date', 'sp', 'CCGT', 'oil', 'coal', 'nuclear', 'wind', 'ps', 'hydro', 'OCGT', 'other', 'intfr', 'intirl', 'intned', 'intew', 'biomass', 'intnem']
data2  = pd.read_csv('GenerationbyFuelType_20200301.csv', names = col2, header = 0, usecols = col2[1:18])
data21 = pd.read_csv('GenerationbyFuelType_20190301.csv', names = col2, header = 0, usecols = col2[1:18])
data22 = pd.read_csv('GenerationbyFuelType_20180301.csv', names = col2, header = 0, usecols = col2[1:18])

data2['date'] = data2['date'].astype(str) + data2['sp'].astype(str)
data2.dropna(axis = 0, inplace = True)
data2.date = data2.date.str.slice(0, -2).astype(int)
data2.set_index(data2.date, inplace = True)
data2.drop(['sp', 'date'], axis = 1, inplace = True)
data2.index.name = 'YYYY/MM/DD/SP'

data21['date'] = data21['date'].astype(str) + data21['sp'].astype(str)
data21.dropna(axis = 0, inplace = True)
data21.date = data21.date.str.slice(0, -2).astype(int)
data21.set_index(data21.date, inplace = True)
data21.drop(['sp', 'date'], axis = 1, inplace = True)
data21.index.name = 'YYYY/MM/DD/SP'

data22['date'] = data22['date'].astype(str) + data22['sp'].astype(str)
data22.dropna(axis = 0, inplace = True)
data22.date = data22.date.str.slice(0, -2).astype(int)
data22.set_index(data22.date, inplace = True)
data22.drop(['sp', 'date'], axis = 1, inplace = True)
data22.index.name = 'YYYY/MM/DD/SP'

w = 0.2
plt.bar(np.arange(len(data2.columns)) + w, data2.mean(), w, color = 'lightblue', label = '2020')
plt.bar(np.arange(len(data2.columns)), data21.mean(), w, color = 'yellow', label = '2019')
plt.bar(np.arange(len(data2.columns)) - w, data22.mean(), w, color = 'green', label = '2018')

plt.xticks(np.arange(len(data2.columns)), data2.columns, rotation = 90)
plt.xlabel('Different types of generation')
plt.ylabel('MW')
plt.title('Mean generation in MW for each type of fuel on the first of March for the year of 2018, 2019 and 2020')
plt.minorticks_on()
plt.legend()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
