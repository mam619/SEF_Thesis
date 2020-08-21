# =============================================================================
# # =============================================================================
# # Plot FS results form Linear Regression onto Polynomial Regression
# # =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# import data & treat it
# =============================================================================
data = pd.read_csv('Results_Random_Forest_Regression_FS.csv', index_col = 0)
data.reset_index(inplace = True)

rmse_gen = data.rmse_general
rmse_gen = rmse_gen.round(1)
rmse_spi = data.rmse_spike
rmse_spi = rmse_spi.round(1)
rmse_nor = data.rmse_normal.round(1)

fontsize = 18


# RESULTS ON ALL TEST SET
plt.figure(figsize = (14,8))

plt.subplot(3, 1, 1)
plt.plot(np.arange(1, (14)), rmse_gen, label = 'RMSE on the total test set', linewidth = 1.2)
plt.xlim(0.5, 13.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
#plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], [], fontsize = fontsize)
plt.ylim(30, 50)
plt.xlim(1,13)
plt.yticks(fontsize = fontsize)
plt.title('Feature Selection results on all regions', fontsize = fontsize + 3)
plt.legend(loc = 'upper right', fontsize = fontsize)


plt.subplot(3, 1, 2)
plt.plot(np.arange(1, (14)), rmse_spi, label = 'RMSE on the spike regions', linewidth = 1.2, color = 'darkorange')
plt.xlim(0.5, 13.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
#plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], [], fontsize = fontsize)
plt.ylim(60.0, 70)
plt.xlim(1,13)
plt.yticks(fontsize = fontsize)
#plt.title('Feature Selection results for Spike regions', fontsize = fontsize + 2)
plt.legend(loc = 'upper right', fontsize = fontsize)


plt.subplot(3, 1, 3)
plt.plot(np.arange(1, (14)), rmse_nor, label = 'RMSE on the normal regions', linewidth = 1.2, color = 'green')
plt.xlim(0.5, 13.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.ylim(20, 45)
plt.xlim(1,13)
#plt.title('Feature Selection results for Normal Regions', fontsize = fontsize + 2)
plt.legend(loc = 'lower right', fontsize = fontsize)
plt.tight_layout()
plt.savefig('Plot_Random_Forest_FS_Results.png')
