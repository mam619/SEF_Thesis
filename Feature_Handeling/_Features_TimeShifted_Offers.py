# =============================================================================
# Lagged price features (time shifted features)
# =============================================================================
# 1 Offer price from previous SP - PrevSP, NO
# 2 Offer price from previous day same SP - PrevDAY
# 3 Offer price from previous week same SP of the week - PrevWEEK
# 4 Offer price from previous month same day and SP - PrevMONTH, NO
# 5 Offer price from last year same day same SP - PrevYEAR, NO
# =============================================================================
# =============================================================================

import pandas as pd

features = pd.read_csv('UK_ARENKO_features.csv', index_col = 0)
offers = pd.read_csv('UK__Offers.csv', index_col = 0)

# combine both offers and features together
data = pd.concat([features, offers], axis=1, sort=True)

# filter any offer higher than 6000 out
offers = offers[offers < 6000]

# fill missing values
data.fillna(value = data.mean(), inplace = True)

# get full set of offers
offers = data['Offers']

# shifts for 1, 2 and 3rd 
shifts = [48, 336]

# create a dictionary of time-shifted data
many_shifts = {'lag_{}'.format(ii): offers.shift(ii) for ii in shifts}

# convert dict into dataframe
many_shifts = pd.DataFrame(many_shifts)

many_shifts.columns = ['PrevDay', 'PrevWeek']

many_shifts.to_csv('UK_shifted_offers.csv')
