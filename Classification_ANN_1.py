# =============================================================================
# First ann for price spike prediction
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import data
data = pd.read_csv('Data_set_1_smaller.csv', index_col = 0)
bin_dataset_1 = pd.read_csv('Spike_binary_1std.csv', index_col = 0)
bin_dataset = bin_dataset_1['spike_occurance']

# filter max values for offer if required
print(data.Offers.max()) #max is 2500... no need to filter max values

# 2017 & 2018 data
data = data.loc[data.index > 2018060000, :]
bin_dataset = bin_dataset.loc[bin_dataset.index > 2018060000]

# reset index
data.reset_index(drop = True, inplace = True)
bin_dataset.reset_index(drop = True, inplace = True)

# Divide features and labels
X = data.iloc[:, 0:15]
y = bin_dataset

# fill nan values
X.fillna(method = 'ffill', inplace = True)

X = X.astype('float64')
X = X.round(20)

# divide data into train and test with 20% test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.1, shuffle=False)

# feature scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# =============================================================================
# ANN design
# =============================================================================

# importing the Keras libraries and packages
import keras
from keras.models import Sequential # to initialise the NN
from keras.layers import Dense # to create layers
from keras import optimizers
from keras import initializers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics

def classifier_tunning(n_hidden = 5, 
                      n_neurons = 40, 
                      optimizer = 'Adamax', 
                      kernel_initializer="he_normal",
                      bias_initializer= initializers.Ones()):
    model = Sequential()
    model.add(Dense(units = n_neurons, 
                    input_dim = 15))
    model.add(keras.layers.LeakyReLU(alpha = 0.2))
    for layer in range(n_hidden):
        model.add(Dense(units = n_neurons))
        model.add(keras.layers.LeakyReLU(alpha = 0.2))
    model.add(Dense(units = 1, 
                    activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer)
    return model

classifier = classifier_tunning()
# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# predicting from y_test
y_pred_prob = classifier.predict(X_test)
y_pred_prob = pd.Series(y_pred_prob[:, 0])

# =============================================================================
# model performance evaluation for different thesholds of confidence
# =============================================================================

from sklearn.preprocessing import binarize

# setup
threshold = np.arange(0.1, 0.4, 0.02)
null_accuracy = []
accuracy_ann = []
f1_ann = []
precision_ann = []
recall_ann = []

spike_accuracy = []
spike_confidence = []

for t in threshold:
    y_pred = binarize([y_pred_prob], t)[0]
    # same as
    # y_pred = y_pred < t
    accuracy_ann.append(metrics.accuracy_score(y_test, y_pred))
    f1_ann.append(metrics.f1_score(y_test, y_pred))
    precision_ann.append(metrics.precision_score(y_test, y_pred))
    recall_ann.append(metrics.recall_score(y_test, y_pred))
    null_accuracy.append(1 - y_test.mean()) # calculate the percentage of zeros = NULL ACCURACY


ann_performance = pd.DataFrame({'threshold':threshold,
                                'null_accuracy':null_accuracy,
                                'accuracy_ann':accuracy_ann,
                                'spike_accuracy':recall_ann, 
                                'spike_confidence':precision_ann,
                                'f1_score': f1_ann})

# NOTE: Null accuracy is achieved by always predicting the most frequent class



