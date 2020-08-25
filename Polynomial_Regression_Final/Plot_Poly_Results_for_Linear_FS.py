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
data = pd.read_csv('Results_Polynomial_Regression_FS.csv', index_col = 0)

rmse_gen = data.rmse_general
rmse_gen = rmse_gen.round(1)
rmse_spi = data.rmse_spike
rmse_spi = rmse_spi.round(1)
rmse_nor = data.rmse_normal.round(1)

fontsize = 18

# RESULTS ON ALL TEST SET
plt.figure(figsize = (14,6))

plt.subplot(3, 1, 1)
plt.plot(np.arange(1, (14)), rmse_gen, label = 'RMSE on the whole test set', linewidth = 2)
plt.xlim(1, 13)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
#plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.ylim(28, 31.5)
plt.title('Polynomial Regression: RMSE for different features set', fontsize = fontsize + 2)
plt.legend(loc = 'upper left', fontsize = fontsize)


plt.subplot(3, 1, 2)
plt.plot(np.arange(1, (14)), rmse_spi, label = 'RMSE on the spike regions', linewidth = 2, color = 'darkorange')
plt.xlim(1, 13)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
#plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.ylim(60, 63)
plt.legend(loc = 'lower right', fontsize = fontsize)


plt.subplot(3, 1, 3)
plt.plot(np.arange(1, (14)), rmse_nor, label = 'RMSE on the non - spike regions', linewidth = 2, color = 'green')
plt.xlim(1, 13)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13], fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.ylim(17.5, 23)
plt.legend(loc = 'upper left', fontsize = fontsize)
plt.tight_layout()
plt.savefig('Plot_Poly_Results_from_Linear_FS.png')
