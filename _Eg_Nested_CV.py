# =============================================================================
# Nested CV search with Medium Tutorial
# =============================================================================

# From skpearn API
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([51, 23, 13, 124, 25, 36])
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# Example of Nested CV from Medium tutorial
df = pd.read_csv('Data\Gemini_ETHUSD_d.csv', skiprows = 1)

# drop nan values
df = df.dropna()

# predict y - open values
X = df.iloc[:, 3:]
y = df.iloc[:, 2]

# split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.4, shuffle=False)

# define the model  -> elastic nets
def build_model(_alpha, _l1_ratio):
    estimator = ElasticNet(
        alpha=_alpha,
        l1_ratio=_l1_ratio,
        fit_intercept=True,
        normalize=False,
        precompute=False,
        max_iter=16,
        copy_X=True,
        tol=0.1,
        warm_start=False,
        positive=False,
        random_state=None,
        selection='random'
    )

    return MultiOutputRegressor(estimator, n_jobs=4)

# do kfold validation to find average scores with time series SPLITTER
model = build_model(_alpha=1.0, _l1_ratio=0.3)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=r2)

# FOR GRIDSEARCH CV do the dame with cv = time series split!

# find the best parameters after fitting X_train with y_train in estimator
# optimal model
model = build_model(_alpha=0.1, _l1_ratio=0.1)
# train model
model.fit(X_train, y_train)
# test score
y_predicted = model.predict(X_test)
score = r2_score(y_test, y_predicted, multioutput='uniform_average')
print("Test Loss: {0:.3f}".format(score))