# =============================================================================
# First ann for price spike prediction
# =============================================================================

import pandas as pd
import numpy as np
from _Ex_Spike_Classification_BinaryDataSet_1std import data18
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data18.iloc[:, 1:-6]
y = data18['spike_occurance']

# divide data into train and test with 10% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.4, shuffle=False)

# feature scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# =============================================================================
# ANN design
# =============================================================================

# importing the Keras libraries and packages
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras.layers import Dropout

# initialises
classifier = Sequential() 

# first layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))
# classifier.add(Dropout(p = 0.1))

# second layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the last output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# compiling the ANN
# For Categorical predictions, loss = categorical underscore cross entropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# predicting from y_test
y_pred_prob = classifier.predict(X_test)
y_pred_prob = pd.Series(y_pred_prob[:, 0])

# =============================================================================
# model performance evaluation for different thesholds of confidence
# =============================================================================
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import binarize
# setup
threshold = np.arange(0.1, 0.4, 0.02)
null_accuracy = []
accuracy_ann = []
spike_accuracy = []
spike_confidence = []

for t in threshold:
    y_pred = binarize([y_pred_prob], t)[0]
    # same as
    # y_pred = y_pred < t
    accuracy_ann.append(accuracy_score(y_test, y_pred))
    null_accuracy.append(1 - y_test.mean()) # calculate the percentage of zeros = NULL ACCURACY
    # calculate confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    spike_accuracy.append(TP/(TP+FN))
    spike_confidence.append(TP/(TN+FP))

ann_performance = pd.DataFrame({'threshold':threshold, 'accuracy_ann':accuracy_ann, 'null_accuracy':null_accuracy, 'spike_accuracy':spike_accuracy, 'spike_confidence':spike_confidence})
# NOTE: Null accuracy is achieved by always predicting the most frequent class



