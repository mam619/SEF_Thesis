# =============================================================================
# Feature manipulation
# 
# All data from 2016 - 01 - 01 to 2018 - 12 - 31
# =============================================================================

import pandas as pd


# =============================================================================
# 1) Previous value Offers to account with seasonality
#       1 Offer price from previous day same SP - PrevDAY
#       2 Offer price from previous week same SP of the week - PrevWEEK
# =============================================================================

shifts = pd.read_csv('UK_shifted_offers.csv', nrows = 52608)
shifts.set_index('Unnamed: 0', inplace = True)
shifts.index.names = ['index']

# ready to concat

# =============================================================================
# 2) from DA Margin DATA:
#      imbalance of the grid for all SP in MW
#      margin available for all SP in MW
# =============================================================================

# download csv created
DA_margin = pd.read_csv('UK_DA_margin_imb_forecast.csv', index_col = [0], usecols=[1, 2,3])

# set equal index to previous data frame
DA_margin.set_index(shifts.index, inplace = True)

# ready to concat

# =============================================================================
# 3) from MARKET DEPTH DATA:
#       volume of all offers and bids in MW
#       volume of accepted offers and bids in MW
# =============================================================================

# download csv created
market_depth = pd.read_csv('UK_Market_depth_data.csv', usecols = [1,2,3,4,5])

# solve index and set it to data frame
market_depth['index'] = market_depth['index'].astype(int)
market_depth.set_index(shifts.index, inplace = True)
market_depth.drop('index', axis = 1, inplace = True)

# ready to concat

# =============================================================================
# 4) Wind Peaks forecast:
#         Time of the peak
#         Peak generation in MW
#         Total meatered capacity at the peak time in MW
# =============================================================================
        
# download csv created
wind_peak = pd.read_csv('UK_wind_peaks_daily_forecast.csv', usecols = [1,2,3,4])

# convert time into sp
wind_peak['sp_peak'] = (wind_peak['time_peak']/100) * 2 + 1

# create the indexes correspondent to peak moments
wind_peak['index'] = wind_peak['index'].astype(str) + wind_peak['sp_peak'].astype(str)
wind_peak['index'] = wind_peak['index'].astype(float).astype(int)

# set right index
wind_peak.set_index('index', inplace = True)
wind_peak.drop('time_peak', axis = 1, inplace =True)
wind_peak.drop('sp_peak', axis = 1, inplace =True)
wind_peak = shifts.join(wind_peak)
wind_peak.drop('PrevDay', axis = 1, inplace =True)
wind_peak.drop('PrevWeek', axis = 1, inplace =True)

# fill nan value with 0
wind_peak.fillna(value = 0, inplace = True)

# make a binary data set to indicate time of peak
wind_peak_bin = wind_peak['peak(MW)']/wind_peak['peak(MW)']
wind_peak_bin.fillna(value = 0, inplace = True)
wind_peak_bin.rename('wind_peak_bin', inplace = True)

# ready to concat

# =============================================================================
# 5) Exchange_rates_daily
# =============================================================================

# download csv created
rates = pd.read_csv('UK_EURGBP_daily_exchange_rate.csv', usecols = [0,2])

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
rates_full = pd.DataFrame({'daily_exchange_rate': l})
rates_full.set_index(wind_peak.index, inplace = True)

# ready to concat

# =============================================================================
# 6) France features 
#       DA price
#       hourly load forecasted
#       hourly generation forecasted
# =============================================================================

import numpy as np

# download csvs
price_france = pd.read_csv('France_DA_prices.csv', nrows = 26281)
gen_france = pd.read_csv('France_generation_forecast.csv', nrows = 26281)
load_france = pd.read_csv('France_load_forecast.csv', nrows = 26281)

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
# 7) Dinorwig presence (Binary Series when Dinorwig offers have been accepted (1) or not (0))
# =============================================================================

dino_bin = pd.read_csv('UK_Dinorwig_presence.csv')
dino_bin.drop('Unnamed: 0', axis = 1, inplace = True)
dino_bin.set_index(shifts.index, inplace = True)

# ready to concat


# =============================================================================
# COMBINE ALL DATA SETS
# =============================================================================

Features_2 = pd.concat([shifts, market_depth, DA_margin, wind_peak_bin, rates_full, france_data, dino_bin], axis = 1)
Features_2.to_csv('Features_APIs.csv')

Features_1 = pd.read_csv('Features_ARENKO.csv')
Features_1.set_index('Unnamed: 0', inplace = True)
Features_1.index.names = ['index']
Features_1.drop('APXV', axis = 1, inplace = True)

Features_final = pd.concat([Features_1, Features_2], axis = 1)
Features_final.to_csv('Features_full_set.csv')

# values missing for each feature
isna = Features_final.isna().sum()
total = Features_final.count()

print('Percentage of elements missing: \n{}'.format((isna/total)*100))


