import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def my_cross_val_predict(pipeline_, X, y, tscv):

    y_test_complete = None
    y_pred_complete = None

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # confirm shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # fit and predict
        pipeline_.fit(X_train, y_train)
        y_pred = pipeline_.predict(X_test)

        # combine and save all y_test and y_pred splits
        if y_test_complete is None:
            y_test_complete = y_test
            y_pred_complete = y_pred
        else:
            y_test_complete = pd.concat([y_test_complete, y_test], axis=0)
            y_pred_complete = np.append(y_pred_complete, y_pred)

    return y_test_complete, y_pred_complete


def my_cross_val_predict_for_lstm(lstm_regressor, scaler, data, tscv, lstm_params, callbacks=None):

    y_test_complete = None
    y_pred_complete = None

    for train_index, test_index in tscv.split(data):

        # divide train and test data
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]

        # scale train and test data separatly
        data_train = scaler.fit_transform(data_train)
        data_test = scaler.transform(data_test)

        # Divide features and labels
        X_train, y_train = data_train[:, :-1], data_train[:, -1]
        X_test, y_test = data_test[:, :-1], data_test[:, -1]

        # confirm shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # split data into correct shape for RNN
        xtrain, ytrain = list(), list()
        xtest, ytest = list(), list()

        for i in range(lstm_params["steps"], len(y_train)):
            xtrain.append(X_train[i - lstm_params["steps"] : i])
            ytrain.append(y_train[i])
        for i in range(lstm_params["steps"], len(y_test)):
            xtest.append(X_test[i - lstm_params["steps"] : i])
            ytest.append(y_test[i])

        # convert data set into array
        X_train, y_train, X_test, y_test = (
            np.array(xtrain),
            np.array(ytrain),
            np.array(xtest),
            np.array(ytest),
        )

        # fit and predict
        lstm_regressor.fit(
            X_train,
            y_train,
            epochs=lstm_params["epochs"],
            batch_size=lstm_params["batch_size"],
            validation_split=0.2,
            callbacks=callbacks,
            shuffle=False,
            verbose=2,
        )
        lstm_regressor.reset_states()
        y_pred = lstm_regressor.predict(X_test)

        # combine and save all y_test and y_pred splits
        if y_test_complete is None:
            y_test_complete = y_test
            y_pred_complete = y_pred
        else:
            y_test_complete = np.append(y_test_complete, y_test)
            y_pred_complete = np.append(y_pred_complete, y_pred)

    # cannot use inverse function; prices col = 14
    y_pred_complete = (y_pred_complete * scaler.scale_[-1]) + (scaler.center_[-1])
    y_test_complete = (y_test_complete * scaler.scale_[-1]) + (scaler.center_[-1])

    return pd.Series(y_test_complete, index=data.index[-len(y_test_complete) :]), pd.Series(
        y_pred_complete, index=data.index[-len(y_test_complete) :]
    )


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


def plot_scatter(y_test, y_pred, filename, fontsize=13, fig_size=(15, 5), path="results/plots/"):

    y_test_spike, y_pred_spike, y_test_normal, y_pred_normal = split_regions(y_test, y_pred)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    line_points = [min(y_test), y_test[5], max(y_test)]

    fig.set_figwidth(fig_size[0])
    fig.set_figheight(fig_size[1])

    ax1.scatter(y_test, y_pred, label="General region", color="dimgray", s=10)
    ax1.plot(line_points, line_points, ls="--", color="dimgrey", label="1-1 line")
    ax1.set_xlabel("Actual", fontsize=fontsize)
    ax1.set_ylabel("Predicted", fontsize=fontsize)
    ax1.legend(loc="upper right", fontsize=fontsize)
    ax1.set_xlim((40, 370))
    ax1.set_ylim((40, 370))
    ax1.title.set_text("(a)")

    ax2.scatter(y_test_spike, y_pred_spike, label="Spike region", color="darkorange", s=10)
    ax2.plot(line_points, line_points, ls="--", color="dimgrey", label="1-1 line")
    ax2.set_xlabel("Actual", fontsize=fontsize)
    ax2.legend(loc="upper right", fontsize=fontsize)
    ax2.set_xlim((40, 370))
    ax2.set_ylim((40, 370))
    ax2.title.set_text("(b)")

    ax3.scatter(y_test_normal, y_pred_normal, label="Normal region", color="dodgerblue", s=10)
    ax3.plot(line_points, line_points, ls="--", color="dimgrey", label="1-1 line")
    ax3.set_xlabel("Actual", fontsize=fontsize)
    ax3.legend(loc="upper right", fontsize=fontsize)
    ax3.set_xlim((40, 370))
    ax3.set_ylim((40, 370))
    ax3.title.set_text("(c)")

    fig.tight_layout()
    fig.savefig(path + "scatter_" + filename + ".png")