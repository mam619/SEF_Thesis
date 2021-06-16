import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def my_cross_val_predict(regressor, X, y, tscv):

    y_test_complete = None
    y_pred_complete = None

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # confirm shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # fit and predict
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # combine and save all y_test and y_pred splits
    if y_test_complete is None:
        y_test_complete = y_test
        y_pred_complete = y_pred
    else:
        y_test_complete = pd.concat([y_test_complete, y_test], axis=0)
        y_pred_complete = np.append(y_pred_complete, y_pred)

    return y_test_complete, y_pred_complete


def select_region(binary_occurences, y_test, y_pred):

    y_test = np.array(y_test)

    y_test_new = y_test * binary_occurences
    y_pred_new = y_pred * binary_occurences

    y_test_new = y_test_new[~np.isnan(y_test_new)]
    y_pred_new = y_pred_new[~np.isnan(y_pred_new)]

    return y_test_new, y_pred_new


def split_regions(y_test, y_pred):

    # download spike indication binary set
    data = pd.read_csv(
        "spike_analysis/spike_binary.csv", index_col=0, parse_dates=True, usecols=[0, 6]
    ).reset_index()

    # remove repeated index values
    data = data.drop_duplicates(subset="index", keep="first").set_index("index")

    # create array same size as y_test with spike occurences
    y_spike_occ = (
        pd.DataFrame(y_test.copy()).join(data).loc[:, "spike_occurance"].replace(0, np.nan).values
    )

    # select y_pred and y_test only for regions with spikes
    y_test_spike, y_pred_spike = select_region(y_spike_occ, y_test.copy(), y_pred.copy())

    # inverse y_spike_occ so the only normal occurences are chosen
    y_normal_occ = (
        (pd.DataFrame(y_test.copy()).join(data).loc[:, "spike_occurance"].replace(1, np.nan).values)
        - 1
    ) * (-1)

    # select y_pred and y_test only for normal regions
    y_test_normal, y_pred_normal = select_region(y_normal_occ, y_test.copy(), y_pred.copy())

    return y_test_spike, y_pred_spike, y_test_normal, y_pred_normal


def get_results(y_test, y_pred):

    y_test_spike, y_pred_spike, y_test_normal, y_pred_normal = split_regions(y_test, y_pred)

    return {
        "rmse_general": mse(y_test, y_pred, squared=False),
        "mae_general": mae(y_test, y_pred),
        "rmse_spike": mse(y_test_spike, y_pred_spike, squared=False),
        "mae_spike": mae(y_test_spike, y_pred_spike),
        "rmse_normal": mse(y_test_normal, y_pred_normal, squared=False),
        "mae_normal": mae(y_test_normal, y_pred_normal),
    }


def plot_results(
    y_test, y_pred, filename, window_plot=144, fontsize=13, fig_size=(11, 4), path="results/plots/"
):

    # download spike indication binary set
    data = pd.read_csv(
        "spike_analysis/spike_binary.csv", index_col=0, parse_dates=True
    ).reset_index()

    # remove repeated index values
    data = data.drop_duplicates(subset="index", keep="first").set_index("index")

    residual = list(y_test) - y_pred

    plt.figure(figsize=fig_size)

    plt.plot(
        data.index[-window_plot:],
        y_test[-window_plot:],
        label="Outturn",
        linewidth=1.5,
        color="steelblue",
    )

    plt.plot(
        data.index[-window_plot:],
        y_pred[-window_plot:],
        label="Predictions",
        linewidth=1.2,
        color="deepskyblue",
    )

    plt.plot(
        data.index[-window_plot:],
        residual[-window_plot:],
        label="Residual error",
        linewidth=0.8,
        color="slategrey",
    )

    plt.fill_between(
        data.index[-window_plot:],
        data["spike_lowerlim"][-window_plot:],
        data["spike_upperlim"][-window_plot:],
        facecolor="skyblue",
        alpha=0.5,
        label="Not spike regions",
    )

    plt.xlim(data.index[-window_plot], data.index[-1])
    # plt.ylim(-100, 260)

    plt.minorticks_on()
    plt.grid(which="major", linestyle="-", linewidth="0.5")
    plt.grid(which="minor", linestyle=":", linewidth="0.5")

    plt.xlabel("2019", fontsize=fontsize)
    plt.ylabel("£/MWh", fontsize=fontsize)

    plt.xticks(fontsize=fontsize - 1)
    plt.yticks(fontsize=fontsize - 1)

    plt.legend(loc="lower right", fontsize=fontsize - 2)

    plt.tight_layout()
    plt.savefig(path + "predictions_" + filename + ".png")


def plot_scatter(y_test, y_pred, filename, fontsize=13, fig_size=(10, 10), path="results/plots/"):

    y_test_spike, y_pred_spike, y_test_normal, y_pred_normal = split_regions(y_test, y_pred)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    fig.set_figheight(6)
    fig.set_figwidth(18)

    ax1.scatter(
        y_test,
        y_pred,
        label="General region",
        color="dimgray",
    )
    ax1.plot(y_test, y_test, ls="--", color="grey", label="1-1 line")
    ax1.set_xlabel("Actual", fontsize=fontsize)
    ax1.set_ylabel("Outturn", fontsize=fontsize)
    ax1.legend(loc="upper right", fontsize=fontsize)
    ax1.set_xlim((40, 370))
    ax1.set_ylim((40, 370))

    ax2.scatter(
        y_test_spike,
        y_pred_spike,
        label="Spike region",
        color="orange",
    )
    ax2.plot(y_test, y_test, ls="--", color="grey", label="1-1 line")
    ax2.set_xlabel("Actual", fontsize=fontsize)
    ax2.legend(loc="upper right", fontsize=fontsize)
    ax2.set_xlim((40, 370))
    ax2.set_ylim((40, 370))

    ax3.scatter(
        y_test_normal,
        y_pred_normal,
        label="Normal region",
        color="steelblue",
    )
    ax3.plot(y_test, y_test, ls="--", color="grey", label="1-1 line")
    ax3.set_xlabel("Actual", fontsize=fontsize)
    ax3.legend(loc="upper right", fontsize=fontsize)
    ax3.set_xlim((40, 370))
    ax3.set_ylim((40, 370))

    fig.tight_layout()
    fig.savefig(path + "scatter_" + filename + ".png")