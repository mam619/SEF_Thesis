
import pandas as pd

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)
bin_dataset = pd.read_csv('Spike_binary_1std.csv', index_col = 0)
bin_dataset = bin_dataset['spike_occurance']

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values

# 2017 & 2018 data
data = data.loc[data.index > 2018060000, :]
bin_dataset = bin_dataset.loc[bin_dataset.index > 2018060000]

# reset index
data.reset_index(drop = True, inplace = True)
bin_dataset.reset_index(drop = True, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:15]
y = bin_dataset