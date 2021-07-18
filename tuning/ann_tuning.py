import pandas as pd
from datetime import datetime
import time
import pytz
import json
import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import initializers, optimizers

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import sys

sys.path.append("../")
import utils


# define objective function
def ann_tuning_objective(params):
    def get_ann(
        n_hidden, n_neurons, kernel_initializer="he_normal", bias_initializer=initializers.Ones()
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

    regressor = KerasRegressor(
        build_fn=lambda: get_ann(n_hidden=params["n_hidden"], n_neurons=params["n_neurons"]),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=0.2,
        shuffle=False,
        verbose=2,
    )

    # create pipeline
    pipeline = Pipeline([("scaler", RobustScaler()), ("regressor", regressor)])

    # nested cross validation
    tscv = TimeSeriesSplit(n_splits=3, max_train_size=183 * 48, test_size=31 * 48)

    # perform nested cross validation and get results
    y_test, y_pred = utils.my_cross_val_predict(pipeline, X, y, tscv)

    # calculate results
    rmse_general = utils.get_results(y_test, y_pred)["rmse_general"]

    return {
        "loss": rmse_general,
        "status": STATUS_OK,
        "eval_time": time.time(),
    }


if __name__ == "__main__":

    # import data
    data = pd.read_csv("data/processed_data/data_final.csv", index_col=0, parse_dates=True)

    # set prediction window according to the date range required
    data = data.loc[
        (data.index >= datetime(2017, 3, 1, tzinfo=pytz.utc))
        & (data.index < datetime(2018, 1, 1, tzinfo=pytz.utc)),
        :,
    ]

    # Divide features and labels
    y = data.pop("offers")
    X = data

    space = {
        "n_hidden": hp.choice("n_hidden", [1, 2, 3, 4, 5, 6]),
        "n_neurons": hp.uniformint("n_neurons", 13, 200),
        "epochs": hp.uniformint("epochs", 100, 500),
        "batch_size": hp.uniformint("batch_size", 13, 100),
    }

    trials = Trials()

    best_ann = fmin(
        fn=ann_tuning_objective, space=space, algo=tpe.suggest, max_evals=1, trials=trials
    )

    ann_hyperparameters = hyperopt.space_eval(space, best_ann)

    # save trials
    pickle.dump(trials, open("results/ann_trials.p", "wb"))

    # save best results
    with open("results/ann_hyperparameters.json", "w") as f:
        json.dump(ann_hyperparameters, f)
