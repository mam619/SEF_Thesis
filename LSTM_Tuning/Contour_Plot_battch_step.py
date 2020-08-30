# =============================================================================
# Contour plots for steps vs. batch_size for LSTM
# =============================================================================

import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

data = pd.read_csv('Results_LSTM_steps_batch.csv')

x = [48, 96, 336]
y = [48, 96, 336]

# X and Y matrix
X, Y = np.meshgrid(x, y)

# creating the 3rd matrix Z for all regions
A = data.rmse_general
A = np.array(A)
Z = np.reshape(A, (len(x), len(y))).T

A_ = data.rmse_spike
A_ = np.array(A_)
Z_ = np.reshape(A_, (len(x), len(y))).T

A__ = data.rmse_normal
A__ = np.array(A__)
Z__ = np.reshape(A__, (len(x), len(y))).T
    

size = 14
fontsize = 15

plt.figure(figsize = (15.5,4.25))
plt.subplot(1, 3, 1)
plt.contourf(X, Y, Z, 10, cmap = 'Blues')
cd = plt.colorbar()
cd.set_label(label = 'RMSE (£/MWh)', size = fontsize)
cd.ax.tick_params(labelsize = fontsize)
plt.xlabel('Steps', fontsize = fontsize)
plt.ylabel('Batch size', fontsize = fontsize)
plt.scatter([48, 48, 48],[48, 96, 336], color = 'black', s = 10)
plt.scatter([96, 96, 96],[48, 96, 336], color = 'black', s = 10)
plt.scatter([336, 336, 336],[48, 96, 336], color = 'black', s = 10)
plt.xticks([48, 96, 336], fontsize = fontsize)
plt.yticks([48, 96, 336], fontsize = fontsize)
plt.title('    Results for the whole test set\n', fontsize = fontsize + 2)


plt.subplot(1, 3, 2)
plt.contourf(X, Y, Z_, 10, cmap = 'Oranges')
cd = plt.colorbar()
cd.set_label(label = 'RMSE (£/MWh)', size = fontsize)
cd.ax.tick_params(labelsize = fontsize)
plt.xlabel('Steps', fontsize = fontsize)
plt.ylabel('Batch size', fontsize = fontsize)
plt.scatter([48, 48, 48],[48, 96, 336], color = 'black', s = 10)
plt.scatter([96, 96, 96],[48, 96, 336], color = 'black', s = 10)
plt.scatter([336, 336, 336],[48, 96, 336], color = 'black', s = 10)
plt.xticks([48, 96, 336], fontsize = fontsize)
plt.yticks([48, 96, 336], fontsize = fontsize)
plt.title('  Results for spike regions\n', fontsize = fontsize + 2)


plt.subplot(1, 3, 3)
plt.contourf(X, Y, Z__, 10, cmap = 'Greens')
cd = plt.colorbar()
cd.set_label(label = 'RMSE (£/MWh)', size = fontsize)
#cd.set_ticks([20.8, 22.4, 24.0, 25.6, 27.2])
cd.ax.tick_params(labelsize = fontsize)
plt.xlabel('Steps', fontsize = fontsize)
plt.ylabel('Batch size', fontsize = fontsize)
plt.scatter([48, 48, 48],[48, 96, 336], color = 'black', s = 10)
plt.scatter([96, 96, 96],[48, 96, 336], color = 'black', s = 10)
plt.scatter([336, 336, 336],[48, 96, 336], color = 'black', s = 10)
plt.xticks([48, 96, 336], fontsize = fontsize)
plt.yticks([48, 96, 336], fontsize = fontsize)
plt.title('   Results for non-spike regions\n', fontsize = fontsize + 2)

plt.tight_layout()
plt.show()

plt.savefig('Contour_plot_Batch_size_Steps.png')

