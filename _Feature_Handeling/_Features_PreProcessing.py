# =============================================================================
# # =============================================================================
# # Data Pre-processing & Visualisation 
# # =============================================================================
# =============================================================================

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# download and treat data
# =============================================================================
Features_final = pd.read_csv('Features_full_set.csv', index_col = 0)
offers = pd.read_csv('UK__Offers.csv', index_col = 0)

# limit offers to 3000
offers = offers[offers < 3000]

# combine both offers and features together
data = pd.concat([Features_final, offers], axis=1, sort=True)
# not interested in DRM feature so drop it
data.drop('DRM', axis = 1, inplace = True)

# =============================================================================
# quantify nan values 
# =============================================================================
print('Missing values (%):\n {}'.format((data.isna().sum()/data.count())*100))

# =============================================================================
# CORR MATRIX - 0
# =============================================================================
corr_matrix_0 = data.corr()

# =============================================================================
# MAKE DESIRED CHANGES
# =============================================================================

# create ratio of accepted offer/bid to total volume of both
data.drop('Offers', axis = 1, inplace = True)
data['ratio_offers_vol'] = data['Accepted_offer_vol']/data['Offer_vol']
data['ratio_bids_vol'] = data['Accepted_bid_vol']/data['Bid_vol']

data.drop('Accepted_offer_vol', axis = 1, inplace = True)
data.drop('Offer_vol', axis = 1, inplace = True)
data.drop('Accepted_bid_vol', axis = 1, inplace = True)
data.drop('Bid_vol', axis = 1, inplace = True)

# create new feature in france with Load - Generation (due to high correlation between both)
data['DA_imb_France'] = data['load_france_forecast']/data['gen_france_forecast']
data.drop('load_france_forecast', axis = 1, inplace = True)
data.drop('gen_france_forecast', axis = 1, inplace = True)

data = pd.concat([data, offers], axis=1, sort=True)

# =============================================================================
# CORR MATRIX - 1
# =============================================================================
corr_matrix_1 = data.corr()

# =============================================================================
# SHIFT features accordingly to problems need
# =============================================================================
# forecasted features & offers shift -3 
# real time values shift + 1
data['Offers'] = data['Offers'].shift(-3)
data['PrevDay'] = data['PrevDay'].shift(-3)
data['PrevWeek'] = data['PrevWeek'].shift(-3)
data['APXP'] = data['APXP'].shift(-3)
data['Rene'] = data['Rene'].shift(-3)
data['TSDF'] = data['TSDF'].shift(-3)
data['LOLP'] = data['LOLP'].shift(-3)
data['DA_margin'] = data['DA_margin'].shift(-3)
data['DA_imb'] = data['DA_imb'].shift(-3)
data['wind_peak_bin'] = data['wind_peak_bin'].shift(-3)
data['daily_exchange_rate'] = data['daily_exchange_rate'].shift(-3)
data['DA_price_france'] = data['DA_price_france'].shift(-3)
data['DA_imb_France'] = data['DA_imb_France'].shift(-3)


data['Ren_R'] = data['Ren_R'].shift(1)
data['NIV'] = data['NIV'].shift(1)
data['Im_Pr'] = data['Im_Pr'].shift(1)
data['In_gen'] = data['In_gen'].shift(1)
data['ratio_offers_vol'] = data['ratio_offers_vol'].shift(1)
data['ratio_bids_vol'] = data['ratio_bids_vol'].shift(1)
data['dino_bin'] = data['dino_bin'].shift(1)

# quantify nan values again
print('Missing values (%):\n {}'.format((data.isna().sum()/data.count())*100))

# =============================================================================
# CORR MATRIX - 2
# =============================================================================
corr_matrix_2 = data.corr()

# =============================================================================
# DELTA CORR MATRIX - changes in both correlation matrices
# =============================================================================
delta_matrix = corr_matrix_1 - corr_matrix_2

# =============================================================================
# SAVE data set 1 
# =============================================================================
data.to_csv('Data_set_1.csv')

# =============================================================================
# PLOTTING
# =============================================================================
# understand distribution of data from non binary Data Sets
data_nonbin = data.loc[:,['Ren_R','APXP','Rene', 'TSDF', 'NIV', 'Im_Pr', 'In_gen','ratio_offers_vol', 'ratio_bids_vol', 'DA_margin', 'DA_imb', 'daily_exchange_rate', 'DA_price_france', 'DA_imb_France', 'Offers']]
data_nonbin.hist(bins = 100, figsize = (30, 15))
plt.savefig('Distribution_plots.png')

# scatter mattrix - BIG MESS
# scatter_matrix(data, figsize = (20, 15))

# =============================================================================
# PLOT CORRELATION ANALYSIS RESULTS
# =============================================================================

values = corr_matrix_2.iloc[-1,:-1]

labels = ['Renewable Ratio',
          'Market Price',
          'Renewable Generation',
          'Transmission Demand',
          'Net Imbalance',
          'Imbalance Price',
          'Interconnectors',
          'Loss of Load Probability',
          'Previous Day SP',
          'Previous Week SP',
          'Grid Margin',
          'Grid Imbalance',
          'Wind Peak Time',
          'Exchange Rate',
          'France Market Price',
          'Dinorwig Plant Presence',
          'Ratio of Offers',
          'Ratio of Bids',
          'France Imbalance']

fontsize = 12

fig = plt.figure(figsize=(11.5, 5))
plt.bar(np.arange(1, 20), abs((values)), 
        edgecolor = 'black',
        linewidth = 1.2)
ax = plt.gca()
ax.set_facecolor('lightsteelblue')
plt.plot(np.arange(0.5, 20.5), np.ones(20) * 0.05, linewidth = 0.8, linestyle = 'dashed', color = 'black')
plt.ylabel('Correlation with the output', fontsize = fontsize)
plt.xticks(np.arange(1, 20), labels, rotation = 80, fontsize = fontsize )
plt.yticks(fontsize = fontsize)
plt.ylim(0,0.4)
plt.xlim(0.5, 19.5)
plt.tight_layout()
plt.savefig('Correlation_analysis_w_output.png')


# =============================================================================
# Data set TOO BIG - filter out features with coeff lower than 0.05 w/ output
# =============================================================================

# wind peak binary is very uncorrelated with all features
data.drop('wind_peak_bin', axis = 1, inplace = True)
# little correlated with Offers
data.drop('daily_exchange_rate', axis = 1, inplace = True)
data.drop('NIV', axis = 1, inplace = True)
data.drop('DA_imb', axis = 1, inplace = True)
data.drop('ratio_bids_vol', axis = 1, inplace = True)

# fourth correlation matrix 
corr_matrix_3 = data.corr()

data.to_csv('Data_set_1_smaller_(1).csv')
