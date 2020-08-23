# =============================================================================
# =============================================================================
# # Plot all results of RMSE for different predictive windows
# =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_lin = pd.read_csv('Results_Linear_Regression_predictive_window.csv', index_col = 0)
data_pol = pd.read_csv('Results_Polynomial_Regression_Predictive_window.csv', index_col = 0)
data_for = pd.read_csv('Results_Random_Forest_Predictive_Window.csv', index_col = 0)
data_svm = pd.read_csv('Results_SVM_Predictive_window.csv', index_col = 0)
data_ann = pd.read_csv('Results_ANN_predictive_window.csv', index_col = 0)
#data_lstm

# RMSE for linear regression
gen_lin = data_lin.rmse_general
spi_lin = data_lin.rmse_spike
nor_lin = data_lin.rmse_normal

# RMSE for poly regression
gen_pol = data_pol.rmse_general
spi_pol = data_pol.rmse_spike
nor_pol = data_pol.rmse_normal

# RMSE for random forest regression
gen_for = data_for.rmse_general
spi_for = data_for.rmse_spike
nor_for = data_for.rmse_normal

# RMSE for svm regression
gen_svm = data_svm.rmse_general
spi_svm = data_svm.rmse_spike
nor_svm = data_svm.rmse_normal

# RMSE for ann regression
gen_ann = data_ann.rmse_general
spi_ann = data_ann.rmse_spike
nor_ann = data_ann.rmse_normal

# =============================================================================
# # RMSE for LSTM regression
# gen_lin = data_lin.rmse_general
# spi_lin = data_lin.rmse_spike
# nor_lin = data_lin.rmse_normal
# =============================================================================

# =============================================================================
# Plotting
# =============================================================================

fontsize = 35
line_width = 7
color1 = 'dodgerblue'
color2 = 'darkorange'
color3 = 'forestgreen'

dates_labels = ['24 ', 
                '22 ',
                '20 ',  
                '18 ', 
                '16 ', 
                '14 ', 
                '12 ',
                '10 ',
                '8 ',
                '6 ',
                '4 ',
                '2 ']


# RESULTS ON ALL TEST SET
plt.figure(figsize = (25,15))

plt.subplot(3, 1, 1)
plt.plot(list(range(12)), gen_lin, label = 'Linear Regression', linewidth = line_width, color = color1)
plt.plot(list(range(12)), gen_pol, label = 'Polynomial Regression', linewidth = line_width, linestyle = 'dotted', color = color1)
plt.plot(list(range(12)), gen_for, label = 'Random Forest', linewidth = line_width, linestyle = (0, (1,10)), color = color1)
plt.plot(list(range(12)), gen_svm, label = 'SVM', linewidth = line_width, linestyle = 'dashed', color = color1)
plt.plot([0, 1, 3, 4, 6, 7, 9, 10], gen_ann, label = 'ANN', linewidth = line_width, linestyle = (0, (3, 1, 1, 1, 1, 1)), color = color1)
# for LSTM do style = (0, (3, 5, 1, 5))
plt.xlim(0, 11)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
#plt.xlabel('Predictive window (in months)', fontsize = fontsize)
plt.xticks(list(range(len(gen_lin))), [] , fontsize = fontsize)
plt.yticks(np.linspace(30, 70, 5),fontsize = fontsize)
plt.title('RMSE results for all models using different prediction windows\n', fontsize = fontsize + 5)

plt.subplot(3, 1, 2)
plt.plot(list(range(12)), spi_lin, label = 'Linear Regression', linewidth = line_width, color = color2)
plt.plot(list(range(12)), spi_pol, label = 'Polynomial Regression', linewidth = line_width, linestyle = 'dotted', color = color2)
plt.plot(list(range(12)), spi_for, label = 'Random Forest', linewidth = line_width, linestyle = (0, (1,10)), color = color2)
plt.plot(list(range(12)), spi_svm, label = 'SVM', linewidth = line_width, linestyle = 'dashed', color = color2)
plt.plot([0, 1, 3, 4, 6, 7, 9, 10], gen_ann, label = 'ANN', linewidth = line_width, linestyle = (0, (3, 1, 1, 1, 1, 1)), color = color2)
# for LSTM do style = (0, (3, 5, 1, 5))
plt.xlim(0, 11)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
#plt.xlabel('Predictive window (in months)', fontsize = fontsize)
plt.xticks(list(range(len(gen_lin))), [], fontsize = fontsize)
plt.yticks(np.linspace(30, 130, 5), fontsize = fontsize)
#plt.title('Results on the Spike regions for different prediction windows', fontsize = fontsize + 5)

plt.subplot(3, 1, 3)
plt.plot(list(range(12)), nor_lin, label = 'Linear Regression', linewidth = line_width, color = color3)
plt.plot(list(range(12)), nor_pol, label = 'Polynomial Regression', linewidth = line_width, linestyle = 'dotted', color = color3)
plt.plot(list(range(12)), nor_for, label = 'Random Forest', linewidth = line_width, linestyle = (0, (1,10)), color = color3)
plt.plot(list(range(12)), nor_svm, label = 'SVM', linewidth = line_width, linestyle = 'dashed', color = color3)
plt.plot([0, 1, 3, 4, 6, 7, 9, 10], gen_ann, label = 'ANN', linewidth = line_width, linestyle = (0, (3, 1, 1, 1, 1, 1)), color = color3)
# for LSTM do style = (0, (3, 5, 1, 5))
plt.xlim(0, 11)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xlabel('Predictive window (in months)', fontsize = fontsize)
plt.xticks(list(range(len(gen_lin))), dates_labels, fontsize = fontsize)
plt.yticks(np.linspace(10, 70, 6), fontsize = fontsize)
#plt.title('Results on the Normal regions for different prediction windows', fontsize = fontsize + 5)
#          fancybox=True, shadow=True, ncol=5, fontsize = fontsize)

plt.tight_layout()
plt.show()
plt.savefig('Plot_Predictive_window_Results.png')


# =============================================================================
# Make legend
# =============================================================================
color3 = 'black'
plt.figure(figsize = (15,10))
plt.plot(list(range(12)), nor_lin, label = 'Linear Regression', linewidth = line_width, color = color3)
plt.plot(list(range(12)), nor_pol, label = 'Polynomial Regression', linewidth = line_width, linestyle = 'dotted', color = color3)
plt.plot(list(range(12)), nor_for, label = 'Random Forest', linewidth = line_width, linestyle = (0, (1,10)), color = color3)
plt.plot(list(range(12)), nor_svm, label = 'SVM', linewidth = line_width, linestyle = 'dashed', color = color3)
plt.plot([0, 1, 3, 4, 6, 7, 9, 10], gen_ann, label = 'ANN', linewidth = line_width, linestyle = (0, (3, 1, 1, 1, 1, 1)), color = color3)
plt.legend()
# for LSTM do style = (0, (3, 5, 1, 5))
plt.xlim(0, 11)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xlabel('Predictive window (in months)', fontsize = fontsize)
plt.xticks(list(range(len(gen_lin))), dates_labels, fontsize = fontsize)
plt.yticks(np.linspace(10, 70, 6), fontsize = fontsize)
plt.title('Feature Selection results for Normal regions', fontsize = fontsize + 2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5, fontsize = fontsize)
plt.savefig('Legend_all.png')
'''
color3 = 'darkorange'
plt.figure(figsize = (15,10))
plt.plot(list(range(12)), nor_lin, label = 'Linear Regression', linewidth = line_width, color = color3)
plt.plot(list(range(12)), nor_pol, label = 'Polynomial Regression', linewidth = line_width, linestyle = 'dotted', color = color3)
plt.plot(list(range(12)), nor_for, label = 'Random Forest', linewidth = line_width, linestyle = (0, (1,10)), color = color3)
plt.plot(list(range(12)), nor_svm, label = 'SVM', linewidth = line_width, linestyle = 'dashed', color = color3)
plt.plot([0, 1, 3, 4, 6, 7, 9, 10], gen_ann, label = 'ANN', linewidth = line_width, linestyle = (0, (3, 1, 1, 1, 1, 1)), color = color3)
plt.legend()
# for LSTM do style = (0, (3, 5, 1, 5))
plt.xlim(0, 11)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xlabel('Predictive window (in months)', fontsize = fontsize)
plt.xticks(list(range(len(gen_lin))), dates_labels, fontsize = fontsize)
plt.yticks(np.linspace(10, 70, 6), fontsize = fontsize)
plt.title('Feature Selection results for Normal regions', fontsize = fontsize + 2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5, fontsize = fontsize)
plt.savefig('Legend_spike.png')

color3 = 'green'
plt.figure(figsize = (15,10))
plt.plot(list(range(12)), nor_lin, label = 'Linear Regression', linewidth = line_width, color = color3)
plt.plot(list(range(12)), nor_pol, label = 'Polynomial Regression', linewidth = line_width, linestyle = 'dotted', color = color3)
plt.plot(list(range(12)), nor_for, label = 'Random Forest', linewidth = line_width, linestyle = (0, (1,10)), color = color3)
plt.plot(list(range(12)), nor_svm, label = 'SVM', linewidth = line_width, linestyle = 'dashed', color = color3)
plt.plot([0, 1, 3, 4, 6, 7, 9, 10], gen_ann, label = 'ANN', linewidth = line_width, linestyle = (0, (3, 1, 1, 1, 1, 1)), color = color3)
plt.legend()
# for LSTM do style = (0, (3, 5, 1, 5))
plt.xlim(0, 11)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.ylabel('RMSE (£/MWh)', fontsize = fontsize)
plt.xlabel('Predictive window (in months)', fontsize = fontsize)
plt.xticks(list(range(len(gen_lin))), dates_labels, fontsize = fontsize)
plt.yticks(np.linspace(10, 70, 6), fontsize = fontsize)
plt.title('Feature Selection results for Normal regions', fontsize = fontsize + 2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5, fontsize = fontsize)
plt.savefig('Legend_normal.png')
'''