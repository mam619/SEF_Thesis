# =============================================================================
# Exercise on spike prediction
# =============================================================================

# import all libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# download data set
from _Ex_Spike_Prediction_knn import data18

# =============================================================================
# Some initial classification predictions
# =============================================================================

# set X and y
X = data18.iloc[:, 1:-6]
y = data18['spike_occurance']

# divide data into train and test with 10% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.4, shuffle=False)

# feature scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# use KNN method for different k values
k_range = range(1, 26)
scores = []

from sklearn.metrics import accuracy_score

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = pd.Series(knn.predict(X_test))
    scores.append(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

# plot results
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

# =============================================================================
# cunfusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

(y_test == 1).sum() # 549
(y_pred == 1).sum() # 15
# model only has 2 TN (spikes well predicted)

spike_accuracy = cm[1][1]/(y_test == 1).sum() # 0.0013 for window = 6
spike_confidence = cm[1][1]/(y_pred == 1).sum() # 0.13 for window = 6
print('Spike accuracy is {}'.format(spike_accuracy))
print('Spike confidence is {}'.format(spike_confidence))

plt.figure(figsize=(15,5))
plt.plot(offers18)
plt.ylabel('$/MWh')
plt.xlabel('Time')
plt.title('Higher value accepted offers for each SP throughout the year of 2018', fontsize = '16')
plt.xticks(np.linspace(0,len(offers18), 12), ['Jan', 'Fev', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Set', 'Oct', 'Nov', 'Dec'])
