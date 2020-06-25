# =============================================================================
# Notes: BOOK Hands on ML w/ Sklearn & Keras, TF
# =============================================================================

import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

# same as
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28,28]),
    keras.layers.Dense(300, activation = 'relu'),
    keras.layers.Dense(100, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax'),
])

model.summary()

model.layers

hidden1 = model.layers[1]
hidden1.name

model.get_layer('dense') is hidden1


weights, biases = hidden1.get_weights()
weights
weights.shape
biases
biases.shape

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "sgd", # this has a default leraning rate, lr = 0.01
              metrics = ["accuracy"])
# optimizer = keras.optimizers.SGD(lr = ???)

history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))

# history.params
# history.epoch

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0,1) # sets vertical range 0-1
plt.show()

# call fit model again to continue trainning 
model.evaluate(X_test, y_test)

# SAVE MODEL
model.save("my_keras_model.h5")

# LOAD MODEL
model = keras.models.load_model("my_keras_model.h5")

# to save checkpoints - use callbacks
# after building and compiling of the mmodels
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                save_best_only = True)
history = model.fit(X_train, y_train, epochs = 10, callbacks = [checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5")
'''

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

def build_model(n_hidden = 1, n_neurons = 30, learning_rate = 3e-3, input_shape = [8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation = 'relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr = learning_rate)
    model.compile(loss = 'mse', optimizer = optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

hist = keras_reg.fit(X_train, y_train, epochs = 100,
              validation_data = (X_valid, y_valid),
              callbacks = [keras.callbacks.EarlyStopping(patience = 10)])

print(hist.history.keys())
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

mse_test = keras_reg.score(X_test, y_test)
# y_pred = keras.predict(X_new)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [3,4,5,6],
    "n_neurons": [68],
    "learning_rate": [0.002705357795009914],
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter = 10, cv = 3)
rnd_search_cv.fit(X_train, y_train, epochs = 100, 
                  validation_data = (X_valid, y_valid),
                  callbacks = [keras.callbacks.EarlyStopping(patience=10)])

# FIND BEST PARAMETERS

rnd_search_cv.best_params_
rnd_search_cv.best_score_
model = rnd_search_cv.best_estimator_.model






