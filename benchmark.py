# benchmark modelling and plotting

import pandas as pd
from datetime import datetime
import numpy as np
import pytz
import json
import matplotlib.pyplot as plt

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

    # BENCHMARK 4 predictions
    y_pred_4 = y.shift(336)[-len(y_test) :]

    # calculate results
    results = utils.get_results(y_test, y_pred_4)

    # save results
    with open("results/results_benchmark_4.json", "w") as f:
        json.dump(results, f)

    utils.plot_results(
        y_test, y_pred_4, filename="benchmark_4", window_plot=200, fontsize=14, fig_size=(15, 5)
    )

    utils.plot_scatter(y_test, y_pred_4, filename="benchmark_4")

    # PLOT ALL BENCHMARKS
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # download spike indication binary set
    data = (
        pd.read_csv("spike_analysis/spike_binary.csv", index_col=0, parse_dates=True)
        .reset_index()
        .drop_duplicates(subset="index", keep="first")
        .set_index("index")
    )

    # required for plotting
    window_plot = 144
    fontsize = 14

    # 1
    axes[0, 0].plot(
        data.index[-window_plot:],
        y_pred_1[-window_plot:],
        label="Benchmark: Mean",
        linewidth=1.2,
        color="deepskyblue",
    )
    axes[0, 0].plot(
        data.index[-window_plot:],
        y_test[-window_plot:],
        label="Outturn",
        linewidth=1.5,
        color="steelblue",
    )
    residual = list(y_test) - y_pred_1
    axes[0, 0].plot(
        data.index[-window_plot:],
        residual[-window_plot:],
        label="Residual error",
        linewidth=0.8,
        color="slategrey",
    )
    axes[0, 0].fill_between(
        data.index[-window_plot:],
        data["spike_lowerlim"][-window_plot:],
        data["spike_upperlim"][-window_plot:],
        facecolor="skyblue",
        alpha=0.5,
        label="Not spike regions",
    )
    axes[0, 0].set_xlim(data.index[-window_plot], data.index[-1])
    axes[0, 0].minorticks_on()
    axes[0, 0].grid(which="major", linestyle="-", linewidth="0.5")
    axes[0, 0].grid(which="minor", linestyle=":", linewidth="0.5")
    axes[0, 0].set_xlabel("2019", fontsize=fontsize)
    axes[0, 0].set_ylabel("£/MWh", fontsize=fontsize)
    axes[0, 0].tick_params(axis="x", labelsize=fontsize - 1)
    axes[0, 0].tick_params(axis="y", labelsize=fontsize - 1)
    axes[0, 0].legend(loc="lower right", fontsize=fontsize - 2)

    # 2
    axes[0, 1].plot(
        data.index[-window_plot:],
        y_pred_2[-window_plot:],
        label="Benchmark: Previous available SP",
        linewidth=1.2,
        color="deepskyblue",
    )
    axes[0, 1].plot(
        data.index[-window_plot:],
        y_test[-window_plot:],
        label="Outturn",
        linewidth=1.5,
        color="steelblue",
    )
    residual = list(y_test) - y_pred_2
    axes[0, 1].plot(
        data.index[-window_plot:],
        residual[-window_plot:],
        label="Residual error",
        linewidth=0.8,
        color="slategrey",
    )
    axes[0, 1].fill_between(
        data.index[-window_plot:],
        data["spike_lowerlim"][-window_plot:],
        data["spike_upperlim"][-window_plot:],
        facecolor="skyblue",
        alpha=0.5,
        label="Not spike regions",
    )
    axes[0, 1].set_xlim(data.index[-window_plot], data.index[-1])
    axes[0, 1].minorticks_on()
    axes[0, 1].grid(which="major", linestyle="-", linewidth="0.5")
    axes[0, 1].grid(which="minor", linestyle=":", linewidth="0.5")
    axes[0, 1].set_xlabel("2019", fontsize=fontsize)
    axes[0, 1].set_ylabel("£/MWh", fontsize=fontsize)
    axes[0, 1].tick_params(axis="x", labelsize=fontsize - 1)
    axes[0, 1].tick_params(axis="y", labelsize=fontsize - 1)
    axes[0, 1].legend(loc="lower right", fontsize=fontsize - 2)

    # 3
    axes[1, 0].plot(
        data.index[-window_plot:],
        y_pred_3[-window_plot:],
        label="Benchmark: Previous day SP",
        linewidth=1.2,
        color="deepskyblue",
    )
    axes[1, 0].plot(
        data.index[-window_plot:],
        y_test[-window_plot:],
        label="Outturn",
        linewidth=1.5,
        color="steelblue",
    )
    residual = list(y_test) - y_pred_3
    axes[1, 0].plot(
        data.index[-window_plot:],
        residual[-window_plot:],
        label="Residual error",
        linewidth=0.8,
        color="slategrey",
    )
    axes[1, 0].fill_between(
        data.index[-window_plot:],
        data["spike_lowerlim"][-window_plot:],
        data["spike_upperlim"][-window_plot:],
        facecolor="skyblue",
        alpha=0.5,
        label="Not spike regions",
    )
    axes[1, 0].set_xlim(data.index[-window_plot], data.index[-1])
    axes[1, 0].minorticks_on()
    axes[1, 0].grid(which="major", linestyle="-", linewidth="0.5")
    axes[1, 0].grid(which="minor", linestyle=":", linewidth="0.5")
    axes[1, 0].set_xlabel("2019", fontsize=fontsize)
    axes[1, 0].set_ylabel("£/MWh", fontsize=fontsize)
    axes[1, 0].tick_params(axis="x", labelsize=fontsize - 1)
    axes[1, 0].tick_params(axis="y", labelsize=fontsize - 1)
    axes[1, 0].legend(loc="lower right", fontsize=fontsize - 2)

    # 4
    axes[1, 1].plot(
        data.index[-window_plot:],
        y_pred_4[-window_plot:],
        label="Benchmark: Previous week SP",
        linewidth=1.2,
        color="deepskyblue",
    )

    axes[1, 1].plot(
        data.index[-window_plot:],
        y_test[-window_plot:],
        label="Outturn",
        linewidth=1.5,
        color="steelblue",
    )
    residual = list(y_test) - y_pred_4
    axes[1, 1].plot(
        data.index[-window_plot:],
        residual[-window_plot:],
        label="Residual error",
        linewidth=0.8,
        color="slategrey",
    )
    axes[1, 1].fill_between(
        data.index[-window_plot:],
        data["spike_lowerlim"][-window_plot:],
        data["spike_upperlim"][-window_plot:],
        facecolor="skyblue",
        alpha=0.5,
        label="Not spike regions",
    )
    axes[1, 1].set_xlim(data.index[-window_plot], data.index[-1])
    axes[1, 1].minorticks_on()
    axes[1, 1].grid(which="major", linestyle="-", linewidth="0.5")
    axes[1, 1].grid(which="minor", linestyle=":", linewidth="0.5")
    axes[1, 1].set_xlabel("2019", fontsize=fontsize)
    axes[1, 1].set_ylabel("£/MWh", fontsize=fontsize)
    axes[1, 1].tick_params(axis="x", labelsize=fontsize - 1)
    axes[1, 1].tick_params(axis="y", labelsize=fontsize - 1)
    axes[1, 1].legend(loc="lower right", fontsize=fontsize - 2)

    plt.tight_layout()

    fig.savefig("results/plots/predictions_benchmarks.png")
