# random forest (multi-var)

import pandas as pd
import json

import constants_

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

import utils

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

    # create regressor
    regressor = RandomForestRegressor(n_estimators=60)

    # create pipeline with regressor and scaler
    pipeline = Pipeline([("scaler", RobustScaler()), ("regressor", regressor)])

    # nested cross validation
    tscv = TimeSeriesSplit(n_splits=3, max_train_size=183 * 48, test_size=15 * 48)

    # perform nested cross validation and get results
    y_test, y_pred = utils.my_cross_val_predict(pipeline, X, y, tscv)

    # calculate results
    results = utils.get_results(y_test, y_pred)

    # save results
    with open("results/results_random_forest.json", "w") as f:
        json.dump(results, f)

    utils.plot_results(
        y_test,
        y_pred,
        filename="random_forest",
        window_plot=200,
        fontsize=14,
        fig_size=(15, 5),
    )

    utils.plot_scatter(y_test, y_pred, filename="random_forest")
