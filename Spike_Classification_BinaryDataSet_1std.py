# =============================================================================
# # =============================================================================
# # Spike occurences definition (Binary data set) for different windows
# # =============================================================================
# =============================================================================

import pandas as pd
import numpy as np

# import data
data = pd.read_csv('Data_set_1.csv', index_col = 0)

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values

# fill nan values
data.fillna(method = 'ffill', inplace = True)

# save data set for binary data set with rolling window of 48 SP 
w = 48
data['sma'] = data['Offers'].rolling(window = w).mean()
data['std'] = data['Offers'].rolling(w).std()
data['spike_upperlim'] = data['sma'] + (data['std'])
data['spike_lowerlim'] = data['sma'] - (data['std'])
data['spike_occurance'] = ((data['Offers'] > data['spike_upperlim']) | (data['Offers'] < data['spike_lowerlim'])).astype(np.int)


# SAVE DATA
data_to_save = data.iloc[:,-6:]

data_to_save.to_csv('Spike_binary_1std.csv')


