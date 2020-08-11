# =============================================================================
# # =============================================================================
# # Plot FS form Linear Regression
# # =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# import data & treat it
# =============================================================================
data = pd.read_csv('Results_Linear_Regression_FS.csv', index_col = 0)

rmse_gen = data.rmse_general.round(1)
rmse_spi = data.rmse_spike
rmse_spi = rmse_spi.round(2)
rmse_nor = data.rmse_normal.round(1)

fontsize = 14

# RESULTS ON ALL TEST SET
plt.figure(figsize = (14,8))

plt.subplot(3, 1, 1)
plt.plot(np.arange(1, (14)), rmse_gen, label = 'RMSE on the total test set', linewidth = 1.2)
plt.xlim(0.5, 13.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], fontsize = fontsize)
plt.yticks(np.linspace(rmse_gen.min(), rmse_gen.max(), 5), fontsize = fontsize)
plt.title('Feature Selection results for All test set', fontsize = fontsize + 2)
plt.legend(loc = 'upper right')


plt.subplot(3, 1, 2)
plt.plot(np.arange(1, (14)), rmse_spi, label = 'RMSE on the spike regions', linewidth = 1.2, color = 'darkorange')
plt.xlim(0.5, 13.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], fontsize = fontsize)
plt.yticks(np.linspace(rmse_spi.min(), rmse_spi.max(), 5), fontsize = fontsize)
plt.title('Feature Selection results for Spike regions', fontsize = fontsize + 2)
plt.legend(loc = 'upper right')


plt.subplot(3, 1, 3)
plt.plot(np.arange(1, (14)), rmse_nor, label = 'RMSE on the normal regions', linewidth = 1.2, color = 'green')
plt.xlim(0.5, 13.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], fontsize = fontsize)
plt.yticks(np.linspace(rmse_nor.min(), rmse_nor.max(), 5), fontsize = fontsize)
plt.title('Feature Selection results for Normal Regions', fontsize = fontsize + 2)
plt.legend(loc = 'lower right')
plt.tight_layout()
plt.savefig('Plot_Linear_Regression_FS_Results.png')
