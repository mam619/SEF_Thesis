# =============================================================================
# # =============================================================================
# # Plot FS form SVM
# # =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# import data & treat it
# =============================================================================
data = pd.read_csv('Results_SVM_FS_linear_kernel.csv', index_col = [0])

rmse_gen = data.rmse_general.round(1)
rmse_spi = data.rmse_spike
rmse_spi = rmse_spi.round(1)
rmse_nor = data.rmse_normal.round(1)

fontsize = 16

# RESULTS ON ALL TEST SET
plt.figure(figsize = (14,6))

plt.subplot(3, 1, 1)
plt.plot(np.arange(1,(15)), rmse_gen, label = 'RMSE on the total test set', linewidth = 2)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xlim(1, 14)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14], [], fontsize = fontsize)
plt.yticks(np.linspace(rmse_gen.min(), rmse_gen.max(), 5), fontsize = fontsize)
plt.title('SVM: RMSE results usign different features set', fontsize = fontsize + 2)
plt.legend(loc = 'upper right', fontsize = fontsize)


plt.subplot(3, 1, 2)
plt.plot(np.arange(1,(15)), rmse_spi, label = 'RMSE on the spike regions', linewidth = 2, color = 'darkorange')
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xlim(1, 14)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14],[], fontsize = fontsize)
plt.yticks(np.linspace(rmse_spi.min(), rmse_spi.max(), 5), fontsize = fontsize)
plt.legend(loc = 'upper right', fontsize = fontsize)


plt.subplot(3, 1, 3)
plt.plot(np.arange(1,(15)), rmse_nor, label = 'RMSE on the non - spike regions', linewidth = 2, color = 'green')
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.xlabel('Number of features', fontsize = fontsize)
plt.ylabel('(£/MWh)', fontsize = fontsize)
plt.xlim(1, 14)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14], fontsize = fontsize)
plt.yticks(np.linspace(rmse_nor.min(), rmse_nor.max(), 5), fontsize = fontsize)
plt.legend(loc = 'upper right', fontsize = fontsize)
plt.tight_layout()
plt.savefig('Plot_SVM_FS_Results.png')
