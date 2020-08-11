# =============================================================================
# 3D plot for steps and batch size for LSTM rtesutls
# =============================================================================

from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

fig = plt.figure(figsize = (10,5))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z,
                cmap='winter', 
                edgecolor='none')
ax.set_xlabel('Batch_size', fontsize = 13)
ax.set_ylabel('Steps', fontsize = 13)
ax.set_zlabel('RMSE (£/MWH)', fontsize = 13)
#ax.set_xticks([48, 96, 336])
#ax.set_yticks([48, 96, 336])
ax.set_title('LSTM: Batch size vs. Step size\n on the whole data set', fontsize = 16)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink = 0.7, aspect = 10)
plt.tight_layout()
plt.show()
plt.savefig('Results_LSTM_batch_step_2.png')

fig = plt.figure(figsize = (10,5))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z_,
                cmap='autumn', 
                edgecolor='none')
ax.set_xlabel('Batch_size', fontsize = 13)
ax.set_ylabel('Steps', fontsize = 13)
ax.set_zlabel('RMSE (£/MWH)', fontsize = 13)
#ax.set_xticks([48, 96, 336])
#ax.set_yticks([48, 96, 336])
ax.set_title('LSTM: Batch size vs. Step size\n on the spike regions', fontsize = 16)
fig.colorbar(surf, shrink = 0.7, aspect = 10)
plt.tight_layout()
plt.show()
plt.savefig('Results_LSTM_batch_step_2.png')

fig = plt.figure(figsize = (10,5))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z__,
                cmap='summer', 
                edgecolor='none')
ax.set_xlabel('Batch_size', fontsize = 13)
ax.set_ylabel('Steps', fontsize = 13)
ax.set_zlabel('RMSE (£/MWH)', fontsize = 13)
#ax.set_xticks([48, 96, 336])
#ax.set_yticks([48, 96, 336])
ax.set_title('LSTM: Batch size vs. Step size\n on the normal regions', fontsize = 16)
fig.colorbar(surf, shrink = 0.7, aspect = 10)
plt.tight_layout()
plt.show()
plt.savefig('Results_LSTM_batch_step_2.png')
