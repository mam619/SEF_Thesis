# =============================================================================
# Data Pre-processing & Visualisation 
# =============================================================================

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt

# download data set
Features_final = pd.read_csv('Feature_Handeling/Features_full_set.csv', index_col = 0)
offers = pd.read_csv('Feature_Handeling/UK__Offers.csv', index_col = 0)

# filter any offer higher than 3000 out
offers = offers[offers < 3000]

# combine both offers and features together
data = pd.concat([Features_final, offers], axis=1, sort=True)

# quantify nan values 
print('Missing values:\n {}'.format(data.isna().sum()))

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

# MAKE DESIRED CHANGES
# create ratio of accepted offer/bid to total volume of both
data.drop('Offers', axis = 1, inplace = True)
data['ratio_offers_vol'] = data['Accepted_offer_vol']/data['Offer_vol']
data['ratio_bids_vol'] = data['Accepted_bid_vol']/data['Bid_vol']

data.drop('Accepted_offer_vol', axis = 1, inplace = True)
data.drop('Offer_vol', axis = 1, inplace = True)
data.drop('Accepted_bid_vol', axis = 1, inplace = True)
data.drop('Bid_vol', axis = 1, inplace = True)

data = pd.concat([data, offers], axis=1, sort=True)

# save

# correlation matrix
corr_matrix = data.corr()
# NOTE: Correlation values drop slighly if nan values filled before with median()


# shift features accordingly
# forecasted features & offers shift -3 
data['Offers'] = data['Offers'].shift(-3)
data['PrevDay'] = data['PrevDay'].shift(-3)
data['PrevWeek'] = data['PrevWeek'].shift(-3)
data['APXP'] = data['APXP'].shift(-3)
data['Rene'] = data['Rene'].shift(-3)
data['TSDF'] = data['TSDF'].shift(-3)
data['DRM'] = data['DRM'].shift(-3)
data['LOLP'] = data['LOLP'].shift(-3)
data['DA_margin'] = data['DA_margin'].shift(-3)
data['DA_imb'] = data['DA_imb'].shift(-3)
data['wind_peak_bin'] = data['wind_peak_bin'].shift(-3)
data['daily_exchange_rate'] = data['daily_exchange_rate'].shift(-3)
data['DA_price_france'] = data['DA_price_france'].shift(-3)
data['load_france_forecast'] = data['load_france_forecast'].shift(-3)
data['gen_france_forecast'] = data['gen_france_forecast'].shift(-3)
# actual value shift +1
data['Ren_R'] = data['Ren_R'].shift(1)
data['NIV'] = data['NIV'].shift(1)
data['Im_Pr'] = data['Im_Pr'].shift(1)
data['In_gen'] = data['In_gen'].shift(1)
data['ratio_offers_vol'] = data['ratio_offers_vol'].shift(1)
data['ratio_bids_vol'] = data['ratio_bids_vol'].shift(1)
data['dino_bin'] = data['dino_bin'].shift(1)

# second correlation matrix 
corr_matrix_2 = data.corr()

# changes in both correlation matrices
delta_matrix = corr_matrix - corr_matrix_2

# MAKE DESIRED CHANGES
# create new feature in france with Load - Generation (due to high correlation between both)
data.drop('Offers', axis = 1, inplace = True)
data['DA_imb_France'] = data['load_france_forecast']/data['gen_france_forecast']

data.drop('gen_france_forecast', axis = 1, inplace = True)
data.drop('load_france_forecast', axis = 1, inplace = True)

data = pd.concat([data, offers], axis=1, sort=True)

# third correlation matrix 
corr_matrix_3 = data.corr()
# Im_France became more correlated with Offer than load or generation!

data.to_csv('Data_set_1.csv')
