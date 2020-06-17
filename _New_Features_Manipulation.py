# =============================================================================
# Feature manipulation
# 
# All data from 2016 - 01 - 01 to 2018 - 12 - 31
# =============================================================================

import pandas as pd

# =============================================================================
# 1) from MARKET DEPTH DATA:
#       volume of all offers and bids in MW
#       volume of accepted offers and bids in MW
# =============================================================================

# download csv created
market_depth = pd.read_csv('.UK_Market_Depth_Data.csv', usecols = [1,2,3,4,5])

# solve index and set it to data frame
market_depth['index'] = market_depth['index'].astype(int)
market_depth.set_index('index', inplace = True)

# if there is any nan value fill with mean
market_depth.fillna(value = market_depth.mean(), inplace = True)

# create a ratio of accepted to total number of offers and bids (seperatly)
ratios = pd.DataFrame({'Ratio_offer_volumes': market_depth['Accepted_offer_vol']/market_depth['Offer_vol'],
                       'Ratio_bid_volumes': market_depth['Accepted_bid_vol']/market_depth['Bid_vol']})

# set same index to ratios data frame
ratios.set_index(market_depth.index, inplace = True)

# ready to concat

# =============================================================================
# 2) from DA Margin DATA:
#      imbalance of the grid for all SP in MW
#      margin available for all SP in MW
# =============================================================================

# download csv created
DA_margin = pd.read_csv('.UK_DA_Margin_Imb_FORECAST.csv', index_col = [0], usecols=[1, 2,3])

# if there is any nan value fill with mean
DA_margin.fillna(value = DA_margin.mean(), inplace = True)

# ready to concat

# =============================================================================
# 3) Wind Peaks forecast:
#         Time of the peak
#         Peak generation in MW
#         Total meatered capacity at the peak time in MW
# =============================================================================
        
# download csv created
wind_peak = pd.read_csv('.UK_Wind_peaks_daily_FORECAST.csv', usecols = [1,2,3,4])

# convert time into sp
wind_peak['sp_peak'] = (wind_peak['time_peak']/100) * 2 + 1

# create the indexes correspondent to peak moments
wind_peak['index'] = wind_peak['index'].astype(str) + wind_peak['sp_peak'].astype(str)
wind_peak['index'] = wind_peak['index'].astype(float).astype(int)

# set right index
wind_peak.set_index('index', inplace = True)
wind_peak.drop('time_peak', axis = 1, inplace =True)
wind_peak.drop('sp_peak', axis = 1, inplace =True)
wind_peak = DA_margin.join(wind_peak)
wind_peak.drop('DA_margin', axis = 1, inplace =True)
wind_peak.drop('DA_imb', axis = 1, inplace =True)

# fill nan value with 0
wind_peak.fillna(value = 0, inplace = True)

# ready to concat

# =============================================================================
# 4) Exchange_rates_daily
# =============================================================================

# download csv created
rates = pd.read_csv('.EURGBP_Exchange_rates_daily.csv', usecols = [0,2])

# set index right
rates[['d','m', 'y']] = rates.Date.str.split("/",expand=True)
rates['index'] = rates.y.astype(str) + rates.m.astype(str) + rates.d.astype(str)
rates.set_index('index', inplace = True)
rates.drop('Date', axis = 1, inplace = True)
rates.drop('y', axis = 1, inplace = True)
rates.drop('m', axis = 1, inplace = True)
rates.drop('d', axis = 1, inplace = True)

# reverse rows order
rates = rates.iloc[::-1]

# create full data set with repeated rate for day
rates = rates.EUR.to_list()

# iterate values to create constant rate during the day
l = []
for i in rates:
    c = 0
    while c < 48:
        l.append(i)
        c = c + 1

# create full data set
rates_full = pd.DataFrame({'daily_rate': l})
rates_full.set_index(wind_peak.index, inplace = True)

# ready to concat

