# =============================================================================
# Multiple_Linear_Regression_all_in.py
# =============================================================================
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read data from csv files
offers = pd.read_csv('Offers.csv', index_col=0)
offers = offers[offers < 2000]

X = pd.read_csv('System Features.csv', parse_dates=True, index_col=0)
y = offers['Offers']
# =============================================================================

data = pd.concat([X, offers], axis=1, sort=True)
#offers.isna().sum()
#offers.fillna(offers.median(), inplace=True)
#offers.dropna(inplace=True)
# #offers.max()
# #offers.min()
# #offers.describe()

# Offers from 2018 only
offers_2018 = offers.loc[offers.index > 2018000000, :]
#print(offers.loc[2018050435:2018050442])
#print(offers.iloc[1])

# Data from 2018
data_2018 = data.loc[data.index > 2018000000, :]

# Handle missing data
data_2018.fillna(data_2018.median(), inplace=True)

# Predict 1h ahead instead of same time
data_2018['Offers'] = data_2018['Offers'].shift(-2)
# ffill: propagate last valid observation forward to next valid backfill 
data_2018['Offers'].fillna(method='ffill', inplace=True)
 
# Divide features and output (with shift done and nan filled)
X = data_2018.iloc[:, :-1]
y = data_2018['Offers']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.1, shuffle=False)
 
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)

r=2
mae = round(metrics.mean_absolute_error(y_test, y_pred), r)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), r)
mse = round(metrics.mean_squared_error(y_test, y_pred), r)


# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(y_test[700:900])), y_test[700:900], zorder=1, lw=3,
          color='red', label='Real Price')
axes.plot(y_pred[700:900], zorder=2, lw=3,
          color='blue', label='LR Predicted Price')
axes.plot(y_pred[700:900]-y_test[700:900].values, zorder=0, lw=3,
          color='green', label='Residual Error')
axes.set_title('Multiple Linear Regression, "All-in" Approach',
               fontsize=18)
axes.set_xlabel('Day and SP', fontsize=16)
axes.set_ylabel('Offer price and Residual Error', fontsize=16)
axes.legend(loc='best', fontsize=16)
axes.grid(True)
axes.autoscale()

print('')
print("   ******** Evaluation Metrics ********    ")
print("Mean Absolute Error:")
print(mae)
print("Mean Squared Error:")
print(mse)
print("Root Mean Squared Error:")
print(rmse)

# =============================================================================



