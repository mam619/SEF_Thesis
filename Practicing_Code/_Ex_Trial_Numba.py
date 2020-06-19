# =============================================================================
# Trial with Numba
# =============================================================================
from numba import njit, jit

import math

@jit
def hypot(x, y):
    x = abs(x)
    y = abs(y)
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)

# %timeit hypot.py_func(3.0, 4.0)
    
# %timeit hypot(3.0, 4.0)
    
@njit      # or @jit(nopython=True)
def function():
    a = "done"
    # your loop or numerically intensive computations
    return a

# ANN 
import numpy as np
import pandas as pd

database = pd.read_csv('Data/Churn_Modelling.csv')
X = database.iloc[:, 3:13].values 
Y = database.iloc[:, 13].values

X_trial = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder() # Transform Gender in binary
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
ct = ColumnTransformer([('Country', OneHotEncoder(),[1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float) # Encode countries
X = X[:, 1:] # Dummy variable TRAP !!! remove one of the columns

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - ANN design
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers

@jit
def ANN(a):
    #a = epochs
    classifier = Sequential() # Initialises
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, Y_train, batch_size = 10, epochs = a)
    y_pred = classifier.predict(X_test)
    return y_pred