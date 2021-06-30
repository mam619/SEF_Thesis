# benchmark modelling and plotting

import pandas as pd
from datetime import datetime
import numpy as np
import pytz
import json

from sklearn.model_selection import train_test_split

import utils

if __name__ == "__main__":

    # import data
    data = pd.read_csv("data/processed_data/data_final.csv", index_col=0, parse_dates=True)

    # set prediction window according to the date range required
    data = data.loc[data.index > datetime(2017, 6, 1, tzinfo=pytz.utc), :]

    # Divide features and labels
    y = data.pop("offers")
    X = data

    # divide target into train and test with 33% test data
    y_train, y_test = train_test_split(y, test_size=0.33, shuffle=False)

    # BENCHMARK 1 predictions
    y_pred_1 = np.ones(len(y_test)) * y_train.mean()

    # calculate results
    results = utils.get_results(y_test, y_pred_1)

    # save results
    with open("results/results_benchmark_1.json", "w") as f:
        json.dump(results, f)

    utils.plot_results(
        y_test, y_pred_1, filename="benchmark_1", window_plot=200, fontsize=14, fig_size=(15, 5)
    )

    utils.plot_scatter(y_test, y_pred_1, filename="benchmark_1")

    # BENCHMARK 2 predictions
    y_pred_2 = X.loc[:, "prev_sp_offers"][-len(y_test) :]

    # calculate results
    results = utils.get_results(y_test, y_pred_2)

    # save results
    with open("results/results_benchmark_2.json", "w") as f:
        json.dump(results, f)

    utils.plot_results(
        y_test, y_pred_2, filename="benchmark_2", window_plot=200, fontsize=14, fig_size=(15, 5)
    )

    utils.plot_scatter(y_test, y_pred_2, filename="benchmark_2")

    # BENCHMARK 3 predictions
    y_pred_3 = X.loc[:, "prev_day_offers"][-len(y_test) :]

    # calculate results
    results = utils.get_results(y_test, y_pred_3)

    # save results
    with open("results/results_benchmark_3.json", "w") as f:
        json.dump(results, f)

    utils.plot_results(
        y_test, y_pred_3, filename="benchmark_3", window_plot=200, fontsize=14, fig_size=(15, 5)
    )

    utils.plot_scatter(y_test, y_pred_3, filename="benchmark_3")
