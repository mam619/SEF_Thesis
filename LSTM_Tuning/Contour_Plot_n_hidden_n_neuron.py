# =============================================================================
# Contour plots for n_hidden vs. n_neurons for LSTM
# =============================================================================

import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

data = pd.read_csv('Results_LSTM_2_n_neurons_n_hidden.csv')

x = [1, 2]
y = np.arange(50, 250, 50)

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

plt.figure(figsize = (10,4))
plt.contourf(X, Y, Z, 10, cmap = 'Blues')
plt.colorbar().set_label(label = 'RMSE (£/MWh)', size = 14)
plt.xlabel('N_hidden', fontsize = 14)
plt.ylabel('N_neurons', fontsize = 14)
plt.xticks(x, x, fontsize = 14)
plt.yticks(y, y, fontsize = 14)
plt.title('Results for the whole test set', fontsize = 16)
plt.tight_layout()
plt.show()
#plt.savefig('Contour_plot_ANN_n_neurons_n_hidden_all.png')

plt.figure(figsize = (8,5))
plt.contourf(X, Y, Z_, 10, cmap = 'Oranges')
plt.colorbar().set_label(label = 'RMSE (£/MWh)', size = 14)
plt.xlabel('N_hidden', fontsize = 16)
plt.ylabel('N_neurons', fontsize = 16)
plt.xticks(x, x, fontsize = 14)
plt.yticks(y, y, fontsize = 14)
plt.title('Results for the spike regions', fontsize = 18)
plt.tight_layout()
plt.show()
#plt.savefig('Contour_plot_ANN_n_neurons_n_hidden_spike.png')

plt.figure(figsize = (8,5))
plt.contourf(X, Y, Z__, 10, cmap = 'Greens')
plt.colorbar().set_label(label = 'RMSE (£/MWh)', size = 14)
plt.xlabel('N_hidden', fontsize = 16)
plt.ylabel('N_neurons', fontsize = 16)
plt.xticks(x, x, fontsize = 14)
plt.yticks(y, y, fontsize = 14)
plt.title('Results for the normal regions', fontsize = 18)
plt.tight_layout()
plt.show()
#plt.savefig('Contour_plot_ANN_n_neurons_n_hidden_normal.png')
