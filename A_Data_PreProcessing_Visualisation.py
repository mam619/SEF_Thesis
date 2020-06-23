# =============================================================================
# Data Pre-processing & Visualisation 
# =============================================================================

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt

Features_final = pd.read_csv('Feature_Handeling/Features_full_set.csv', index_col = 0)
offers = pd.read_csv('Feature_Handeling/UK__Offers.csv', index_col = 0)

# filter any offer higher than 3000 out
offers = offers[offers < 3000]

# combine both offers and features together
data = pd.concat([Features_final, offers], axis=1, sort=True)

# find nan values 
print('Before filling missing values:\n {}'.format(data.isna().sum()))

# substitute missing values with mean
#data.fillna(value = data.mean(), inplace = True)

print('After filling missing values:\n {}'.format(data.isna().sum()))

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

# PLOTTING

# understand distribution of data from non binary Data Sets
data_nonbin = data.loc[:,['Ren_R','APXP','Rene', 'TSDF', 'NIV', 'Im_Pr', 'In_gen', 'DRM','Accepted_offer_vol', 'Offer_vol', 'Accepted_bid_vol', 'Bid_vol', 'DA_margin', 'DA_imb', 'daily_exchange_rate', 'DA_price_france', 'gen_france_forecast', 'load_france_forecast', 'Offers']]
data_nonbin.hist(bins = 100, figsize = (30, 15))

# scatter mattrix - BIG MESS
# scatter_matrix(data, figsize = (20, 15))

# correlation matrix
corr_matrix = data.corr()
print('CORRELATION MATRIX: \n{}'.format(corr_matrix))
# NOTE: Correlation values drop slighly if nan values filled before with median()


# MAKE DESIRED CHANGES
data.drop('Offers', axis = 1, inplace = True)
data['ratio_offers_vol'] = data['Accepted_offer_vol']/data['Offer_vol']
data['ratio_bids_vol'] = data['Accepted_bid_vol']/data['Bid_vol']

data.drop('Accepted_offer_vol', axis = 1, inplace = True)
data.drop('Offer_vol', axis = 1, inplace = True)
data.drop('Accepted_bid_vol', axis = 1, inplace = True)
data.drop('Bid_vol', axis = 1, inplace = True)

data = pd.concat([data, offers], axis=1, sort=True)

# save

data.to_csv('Data_set_1.csv')

