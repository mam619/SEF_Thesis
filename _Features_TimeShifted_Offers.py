# =============================================================================
# Lagged price features (time shifted features)
# =============================================================================
# 1 Offer price from previous SP - PrevSP
# 2 Offer price from previous day same SP - PrevDAY
# 3 Offer price from previous week same SP of the week - PrevWEEK
# 4 Offer price from previous month same day and SP - PrevMONTH -> DO
# 5 Offer price from last year same day same SP - PrevYEAR -> DO
# =============================================================================
# =============================================================================

import pandas as pd
from _Ex_Spike_Classification_BinaryDataSet_1std import data

offers = data['Offers']

# shifts for 1, 2 and 3rd 
shifts = [1, 48, 336]

# create a dictionary of time-shifted data
many_shifts = {'lag_{}'.format(ii): offers.shift(ii) for ii in shifts}

# convert dict into dataframe
many_shifts = pd.DataFrame(many_shifts)
