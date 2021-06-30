# perform long-short term memory network modelling (multi-var)

import pandas as pd
from datetime import datetime
import pytz
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import utils


lstm_params = {
    "epochs": 500,
    "validation_split": 0.2,
    "batch_size": 50,
    "steps": 48,
    "n_hidden": 2,
    "units": 100,
    "batch_size": 48,
}


def get_lstm(kernel_initializer="he_uniform", bias_initializer=initializers.Ones()):
    model = Sequential()

    if lstm_params["n_hidden"] == 1:
        model.add(
            LSTM(
                units=lstm_params["units"],
                input_shape=(lstm_params["steps"], lstm_params["features_num"]),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
    if lstm_params["n_hidden"] == 2:
        model.add(
            LSTM(
                units=lstm_params["units"],
                input_shape=(lstm_params["steps"], lstm_params["features_num"]),
                return_sequences=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(
            LSTM(
                units=lstm_params["units"],
                input_shape=(lstm_params["steps"], lstm_params["features_num"]),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

    model.add(Dense(1, activation="linear"))

    optimizer = optimizers.RMSprop()
    model.compile(loss="mse", metrics=["mse", "mae"], optimizer=optimizer)
    return model


if __name__ == "__main__":

    # import data
    data = pd.read_csv("data/processed_data/data_final.csv", index_col=0, parse_dates=True)

    # set prediction window according to the date range required
    data = data.loc[data.index > datetime(2017, 6, 1, tzinfo=pytz.utc), :]

    # add features number to lstm params
    lstm_params["features_num"] = data.shape[1] - 1

    # scaler
    scaler = RobustScaler()

    # nested cross validation
    tscv = TimeSeriesSplit(n_splits=6, max_train_size=365 * 48, test_size=48 * 30)

    # set callbacks
    callbacks = [
        EarlyStopping(patience=50),
        ModelCheckpoint(filepath="model_checkpoint", save_weights_only=True, save_best_only=True),
    ]

    # perform nested cross validation and get results
    y_test, y_pred = utils.my_cross_val_predict_for_lstm(
        get_lstm(), scaler, data, tscv, lstm_params, callbacks
    )

    # calculate results
    results = utils.get_results(y_test, y_pred)

    # save results
    with open("results/results_lstm_robust_scaler.json", "w") as f:
        json.dump(results, f)

    utils.plot_results(
        y_test,
        y_pred,
        filename="lstm_robust_scaler",
        window_plot=200,
        fontsize=14,
        fig_size=(15, 5),
    )

    utils.plot_scatter(y_test, y_pred, filename="lstm_robust_scaler")
