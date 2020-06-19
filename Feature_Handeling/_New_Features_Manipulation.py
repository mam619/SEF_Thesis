# =============================================================================
# Feature manipulation
# 
# All data from 2016 - 01 - 01 to 2018 - 12 - 31
# =============================================================================

import pandas as pd


# =============================================================================
# 1) from DA Margin DATA:
#      imbalance of the grid for all SP in MW
#      margin available for all SP in MW
# =============================================================================

# download csv created
DA_margin = pd.read_csv('.UK_DA_Margin_Imb_FORECAST.csv', index_col = [0], usecols=[1, 2,3])

# if there is any nan value fill with mean
DA_margin.fillna(value = DA_margin.mean(), inplace = True)

# ready to concat

# =============================================================================
# 2) from MARKET DEPTH DATA:
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

ratio_offers = market_depth['Accepted_offer_vol']/market_depth['Offer_vol']
ratio_bids = market_depth['Accepted_bid_vol']/market_depth['Bid_vol']

# create a ratio of accepted to total number of offers and bids (seperatly)
ratios = pd.DataFrame({'Ratio_offer_volumes': ratio_offers,
                       'Ratio_bid_volumes': ratio_bids})

# set same index to ratios data frame
ratios.set_index(DA_margin.index, inplace = True)

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

# =============================================================================
# 5) France features 
#       DA price
#       hourly load forecasted
#       hourly generation forecasted
# =============================================================================

import numpy as np

# download csvs
price_france = pd.read_csv('.France_DA_prices.csv', nrows = 26281)
gen_france = pd.read_csv('.France_Generation_forecast.csv', nrows = 26281)
load_france = pd.read_csv('.France_Load_forecast.csv', nrows = 26281)

# drop first row to start at 2016 01 01 at 1 am (mid night in UK)
price_france.drop(0, axis = 0, inplace = True)
gen_france.drop(0, axis = 0, inplace = True)
load_france.drop(0, axis = 0, inplace = True)

# set date indexes (incomplete) from the dataset
price_france.set_index(price_france['Unnamed: 0'].astype(str), inplace = True)
price_france.drop('Unnamed: 0', axis = 1, inplace = True)
gen_france.set_index(gen_france['Unnamed: 0'].astype(str), inplace = True)
gen_france.drop('Unnamed: 0', axis = 1, inplace = True)
load_france.set_index(load_france['Unnamed: 0'].astype(str), inplace = True)
load_france.drop('Unnamed: 0', axis = 1, inplace = True)

# create a complete date index for the range of data used (2016 - 2018)
a = pd.date_range(start='1/1/2016 01:00', end = '01/01/2019 00:00', tz='Europe/Brussels', freq = 'H')
dummy = np.arange(0, 26304)
df = pd.DataFrame(dummy, index = a.astype(str))

# join data frames with dummy dataframe to find nan values
price_france = price_france.join(df, how = 'right')
price_france.columns = ['DA_price_france', 'dummy']
price_france.drop('dummy', axis = 1, inplace = True)

gen_france = gen_france.join(df, how = 'right')
gen_france.columns = ['gen_france_forecast', 'dummy']
gen_france.drop('dummy', axis = 1, inplace = True)

load_france = load_france.join(df, how = 'right')
load_france.columns = ['load_france_forecast', 'dummy']
load_france.drop('dummy', axis = 1, inplace = True)

# fill nan values with mean of all dataset
price_france['DA_price_france'].fillna(price_france['DA_price_france'].mean(), inplace = True)
gen_france['gen_france_forecast'].fillna(gen_france['gen_france_forecast'].mean(), inplace = True)
load_france['load_france_forecast'].fillna(load_france['load_france_forecast'].mean(), inplace = True)


# iterate values to create constant values during the SPs
l = []
for i in price_france['DA_price_france']:
    c = 0
    while c < 2:
        l.append(i)
        c = c + 1

m = []
for i in gen_france['gen_france_forecast']:
    c = 0
    while c < 2:
        m.append(i)
        c = c + 1

n = []
for i in load_france['load_france_forecast']:
    c = 0
    while c < 2:
        n.append(i)
        c = c + 1

# attach to data frame 
france_data = pd.DataFrame({'index': wind_peak.index, 'DA_price_france': l, 'gen_france_forecast': m, 'load_france_forecast': n })
france_data.set_index('index', inplace = True)

# ready to concat

# =============================================================================
# 6) Dinorwig presence (Binary Series when Dinorwig offers have been accepted (1) or not (0))
# =============================================================================

dino_bin = pd.read_csv('.UK_Dinorwig_presence.csv')
dino_bin.set_index('Unnamed: 0', inplace = True)

# ready to concat

# =============================================================================
# COMBINE ALL DATA SETS
# =============================================================================

Features_22 = pd.concat([DA_margin, wind_peak, rates_full, france_data, dino_bin], axis = 1)
Features_2 = pd.concat([DA_margin, wind_peak, rates_full, france_data, ratios, dino_bin], axis = 1)

Features_2.to_csv('.FEATURES_2.csv')