import pandas as pd
from datetime import datetime
import time
import pytz
import json
import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU
from tensorflow.keras import initializers, optimizers


import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import sys

sys.path.append("../")
import utilscolab
import constants_


# define objective function
def lstm_tuning_objective(params):
    def get_lstm(
        n_hidden=params["n_hidden"],
        n_neurons=params["n_neurons"],
        steps=params["steps"],
        kernel_initializer="he_normal",
        bias_initializer=initializers.Ones(),
    ):

        model = Sequential()

        if n_hidden == 1:
            model.add(
                LSTM(
                    units=n_neurons,
                    input_shape=(steps, 14),
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                )
            )
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.2))
        if n_hidden == 2:
            model.add(
                LSTM(
                    units=n_neurons,
                    input_shape=(steps, 14),
                    return_sequences=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                )
            )
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.2))
            model.add(
                LSTM(
                    units=n_neurons,
                    input_shape=(steps, 14),
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

    # scaler
    scaler = RobustScaler()

    # nested cross validation
    tscv = TimeSeriesSplit(n_splits=3, max_train_size=183 * 48, test_size=15 * 48)

    # perform nested cross validation and get results
    y_test, y_pred = utilscolab.my_cross_val_predict_for_lstm(
        get_lstm(), scaler, data, tscv, lstm_params=params
    )

    # calculate results
    rmse_general = utilscolab.get_results(y_test, y_pred)["rmse_general"]

    return {
        "loss": rmse_general,
        "status": STATUS_OK,
        "eval_time": time.time(),
    }


if __name__ == "__main__":

    # import data
    data = pd.read_csv("../data/processed_data/data_final.csv", index_col=0, parse_dates=True)

    # set prediction window according to the date range required
    data = data.loc[
        (data.index >= constants_.TUNING_DATES["start"])
        & (data.index < constants_.TUNING_DATES["end"]),
        :,
    ]

    space = {
        "n_hidden": hp.choice("n_hidden", [1, 2]),
        "n_neurons": hp.uniformint("n_neurons", 13, 150),
        "steps": hp.uniformint("steps", 24, 177),
        "epochs": hp.uniformint("epochs", 100, 500),
        "batch_size": hp.uniformint("batch_size", 13, 100),
    }

    def run_trials():

        trials_step = 10  # how many additional trials to do after loading saved trials. 1 = save after iteration
        max_trials = 20  # initial max_trials. put something small to not have to wait

        try:  # try to load an already saved trials object, and increase the max
            trials = pickle.load(open("lstm_trials_colab_loop.p", "rb"))
            max_trials = len(trials.trials) + trials_step

        except:  # create a new trials object and start searching
            trials = Trials()

        best_lstm = fmin(
            fn=lstm_tuning_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_trials,
            trials=trials,
        )

        # find best set of hyperparameters and add correspondent loss
        lstm_hyperparameters = hyperopt.space_eval(space, best_lstm)
        lstm_hyperparameters["loss"] = float(
            pd.DataFrame(trials.trials)[["result"]].applymap(lambda x: x["loss"]).min()
        )

        # save best results
        with open("lstm_hyperparameters_colab_loop.json", "w") as f:
            json.dump(lstm_hyperparameters, f)

        # save the trials object
        pickle.dump(trials, open("lstm_trials_colab_loop.p", "wb"))

    # loop indefinitely and stop whenever you like
    while True:
        run_trials()
