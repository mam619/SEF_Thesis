# =============================================================================
# Visualisation of data for different time gaps
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FOR 1 STD:

# for window = 50, plot last 24 hours of 2018
w = 50
plt.figure(figsize=(15,5))
plt.plot(data[-48:]['Offers'], label = 'Offer')
plt.plot(data[-48:]['sma'], label = 'Simple moving average')
plt.plot(data[-48:]['spike_upperlim'], label = ' Spike upper limit')
plt.plot(data[-48:]['spike_lowerlim'], label = ' Spike lower limit')
plt.title('Offers with spike limits (using 1std) for the last day of 2018 with a rolling window = {} SP'.format(w))
plt.xticks(np.arange(52565, 52613, 2), np.arange(0, 26))
plt.legend()
plt.ylim(0,220)
plt.ylabel('Offer price in £/MWh')
plt.xlabel('Hours of the day')
plt.minorticks_on() #required for the minor grid
plt.grid(which = 'major', linestyle ='-', linewidth = '0.25', color = 'black')
plt.show()


# FOR 2 STD:

data = pd.read_csv('Spike_binary_2std.csv')

# for window = 50, plot last 24 hours of 2018
w = 50
plt.figure(figsize=(15,5))
plt.plot(data[-48:]['Offers'], label = 'Offer')
plt.plot(data[-48:]['sma'], label = 'Simple moving average')
plt.plot(data[-48:]['spike_upperlim'], label = ' Spike upper limit')
plt.plot(data[-48:]['spike_lowerlim'], label = ' Spike lower limit')
plt.title('Offers with spike limits (using 2std) for the last day of 2018 with a rolling window = {} SP'.format(w))
plt.xticks(np.arange(52565, 52613, 2), np.arange(0, 26))
plt.legend()
plt.ylim(0,220)
plt.ylabel('Offer price in £/MWh')
plt.xlabel('Hours of the day')
plt.minorticks_on() #required for the minor grid
plt.grid(which = 'major', linestyle ='-', linewidth = '0.25', color = 'black')
plt.show()
