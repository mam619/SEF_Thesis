# perform artifitial neural network modelling (multi-var)

import pandas as pd
from datetime import datetime
import pytz
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import initializers, optimizers

import utils
import constants_

ann_params = {
    "batch_size": 23,
    "epochs": 426,
    "n_hidden": 1,
    "n_neurons": 55,
    "loss": 45.665047784686266,
}


def get_ann(
    n_hidden=ann_params["n_hidden"],
    n_neurons=ann_params["n_neurons"],
    kernel_initializer="he_normal",
    bias_initializer=initializers.Ones(),
):
    model = Sequential()

    model.add(
        Dense(
            units=n_neurons,
            input_dim=14,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
    )
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.1))

    for _ in range(n_hidden):
        model.add(
            Dense(
                units=n_neurons,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.1))

    model.add(Dense(units=1, activation="linear"))

    optimizer = optimizers.RMSprop()
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])

    return model


if __name__ == "__main__":

    # import data
    data = pd.read_csv("data/processed_data/data_final.csv", index_col=0, parse_dates=True)

    # set prediction window according to the date range required
    data = data.loc[
        (data.index >= constants_.TEST_DATES["start"])
        & (data.index < constants_.TEST_DATES["end"]),
        :,
    ]

    # Divide features and labels
    y = data.pop("offers")
    X = data

    # create regressor and scaler
    regressor = KerasRegressor(
        build_fn=lambda: get_ann(
            n_hidden=ann_params["n_hidden"], n_neurons=ann_params["n_neurons"]
        ),
        epochs=ann_params["epochs"],
        batch_size=ann_params["batch_size"],
        validation_split=0.2,
        shuffle=False,
        verbose=2,
    )

    # create pipeline
    pipeline = Pipeline([("scaler", RobustScaler()), ("regressor", regressor)])

    # nested cross validation
    tscv = TimeSeriesSplit(n_splits=3, max_train_size=(183 * 48), test_size=15 * 48)

    # perform nested cross validation and get results
    y_test, y_pred = utils.my_cross_val_predict(pipeline, X, y, tscv)

    # calculate results
    results = utils.get_results(y_test, y_pred)

    # save results
    with open("results/results_ann.json", "w") as f:
        json.dump(results, f)

    utils.plot_results(
        y_test, y_pred, filename="ann", window_plot=200, fontsize=14, fig_size=(15, 5)
    )

    utils.plot_scatter(y_test, y_pred, filename="ann")
