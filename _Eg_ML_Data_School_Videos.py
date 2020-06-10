# =============================================================================
# DATA SCHOOL VIDEO SERIES ON ML APPLICATION
# =============================================================================
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target

# =============================================================================
# 4 STEP ML APPLICATION // for model training and prediction
# =============================================================================

# Step 1 - Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)
# random_state -> It will split the data in the same way 4 times

# Set 2 - Instantiate the estimator
knn = KNeighborsClassifier(n_neighbors = 1)

# Step 3 - Fit the model with data ("model training")
knn.fit(X_train, y_train)

# Step 4 - Predict the response for a new observation (out of sample data)
knn.predict([[3, 5, 4, 2]]) # Returns a NumPy array; can predict for multiple observations at once
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)

knn1_pred = knn.predict(X_test)

# USING A DIFFERENT VALUE FOR K
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn5_pred = knn.predict(X_test)

# USING A DIFFERENT CLASSIFICATION MODEL
from sklearn.linear_model import LogisticRegression
logred = LogisticRegression()
logred.fit(X_train, y_train)
y_pred = logred.predict(X_test)

# =============================================================================
# How to choose the best model; How to tune parameters; How to classify performance
# =============================================================================

from sklearn.metrics import accuracy_score
accuracy_y = accuracy_score(y_test, y_pred)
accuracy_knn1 = accuracy_score(y_test, knn1_pred)
accuracy_knn5 = accuracy_score(y_test, knn5_pred)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

accuracy_logred = accuracy_score(y_test, logreg_pred)

k_range = range(1, 26)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

# =============================================================================
#  Data Science in Python
# =============================================================================

import pandas as pd
data = pd.read_csv('Data/Advertising.csv', index_col=0)

import seaborn as sns
sns.pairplot(data, x_vars = ['TV', 'Radio', 'Newspaper'], y_vars = 'Sales', size = 7, aspect = 0.7, kind = 'reg')
plt.show()

feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data['Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.intercept_)
print(linreg.coef_)

zip(feature_cols, linreg.coef_)

y_pred2 = linreg.predict(X_test)

from sklearn.metrics import mean_squared_error as mse
import numpy as np
accuracy_linreg = np.sqrt(mse(y_test, y_pred2))

# =============================================================================
# Cross = Validation
# recomendations:
# K = 10 is generally used
# Stratefied sampling -> response class should be represented with equal proportions in each of the K fold
# =============================================================================
from sklearn.datasets import load_iris
from sklearn import metrics
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_knn = metrics.accuracy_score(y_test, y_pred)

# KFold EXPLAINED
from sklearn.model_selection import KFold
kf = KFold( 5, shuffle = False)

# =============================================================================
# Cross Validations with parameter tuning
# =============================================================================
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
scores.mean()
k_range = range(1, 31)
k_scores = []
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, X, y, cv =10, scoring = 'accuracy')
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross - Validated Accuracy')
plt.show()

# =============================================================================
# Find the best model either KNN model or Logistic regression
# =============================================================================
knn = KNeighborsClassifier(n_neighbors=20)
max_accuracy_knn = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy').mean()

logreg = LogisticRegression()
max_accuracy_logreg = cross_val_score(logreg, X, y, cv = 10, scoring = 'accuracy').mean()

# =============================================================================
# NOTE: Performe feature selection WITHIN the CV procedure
# =============================================================================

# =============================================================================
# CV can be used to tune parameters
# =============================================================================
k_range = range(1, 31)
k_scores = []
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, X, y, cv =10, scoring = 'accuracy')
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross - Validated Accuracy')
plt.show()

from sklearn.model_selection import GridSearchCV
# Define parameters values that should be searched
k_range = range(1, 31)
# Create parameter grid in a dictionary
param_grid = dict(n_neighbors =list(k_range))
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy') # you can set n_jobes = -1 to run computation in parallel
# fit the grid with data
grid.fit(X, y)
import pandas as pd
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
grid_mean_scores = grid.cv_results_['mean_test_score']

plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

grid.best_score_
grid.best_params_
grid.best_estimator_

# =============================================================================
# Searching parameters simultaneously with GRID SEARCH
# =============================================================================
k_range = range(1, 31)
weight_options = ['uniform', 'distance']

param_grid = dict(n_neighbors =list(k_range), weights = weight_options)
# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'accuracy')
grid.fit(X, y)

results_2 = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
grid_mean_scores_2 = grid.cv_results_['mean_test_score']

grid.best_score_
grid.best_params_
grid.best_estimator_

# train the model with the best parameters
knn = KNeighborsClassifier(n_neighbors = 13, weights = 'uniform')
knn.fit(X, y)

knn.predict([[3, 5, 4, 2]])

# =============================================================================
# Searching parameters simultaneously with RANDOM SEARCH CV
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV
param_dist = dict(n_neighbors =list(k_range), weights = weight_options)
rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring ='accuracy', n_iter = 10, random_state = 5)
rand.fit(X, y)
results_3 = pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
grid_mean_scores_3 = rand.cv_results_['mean_test_score']

# although it only has 10 iteration; it finds the best result ANYWAY !!!!
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, return_train_score=False)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)
# most of the times that is the case

# NOTE: start with Grid.. then switch to Randomized with lower values of iter

# =============================================================================
# EVALUATING A CLASSIFICATION MODEL
# Model evaluation metrics:
# Regression: mse, rmse
# Classification: accuracy
# =============================================================================

# read the data into a pandas DataFrame
import pandas as pd
path = 'Data/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(path, header=None, names=col_names)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X = pima[feature_cols]
y = pima.label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class) # 0.6927083333, seems good

# NOTE: Null accuracy is achieved by always predicting the most frequent class
# examine the class distribution of the testing set (using a Pandas Series method)
y_test.value_counts()
# calculate the percentage of ones
y_test.mean() # 0.322
# calculate the percentage of zeros = NULL ACCURACY
1 - y_test.mean() # 0.677

# =============================================================================
# NOTE: classification accuracy - the easiest to understand, does not tell the underlying distribution of response values
# does not tell what "type of error the classifier is making
# =============================================================================
print(metrics.confusion_matrix(y_test, y_pred_class))
# THE ORDER OF Y_TEST AND Y_PRED MATTER! TRUE VALUES FIRST

# =============================================================================
# Basic terminology
# True Positives (TP): we correctly predicted that they do have diabetes
# True Negatives (TN): we correctly predicted that they don't have diabetes
# False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error")
# False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error")
# =============================================================================
print('True:', y_test.values[0:25])
print('Pred:', y_pred_class[0:25])

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# METRICS COMPUTED FROM A CONFUSION MATRIX:
# CLASSIFICATION ACCURACY (how often is it correct)
print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred_class))
# CLASSIFICATION ERROR (how often is it incorrect) - misclassification rate
print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, y_pred_class))
# SPECIFICITY (how often is the prediction correct)
print(TN / float(TN + FP))
# FALSE POSITIVE RATE (how often is prediction incorrect?)
print(FP / float(TN + FP))
# PRECISION (how often is the prediction correct?)
print(TP / float(TP + FP))
print(metrics.precision_score(y_test, y_pred_class))

# =============================================================================
# Many other metrics can be computed: F1 score, Matthews correlation coefficient, etc.
# =============================================================================

# =============================================================================
# Adjusting the classification thershold
# =============================================================================
logreg.predict(X_test)[0:10]
logreg.predict_proba(X_test)[0:10, :]
# predicted probability for class 1
logreg.predict_proba(X_test)[0:10, 1]
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# plot probability for class 1 (having diabetes)
plt.hist(y_pred_prob, bins = 8)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
plt.show()

# Lowering threshold increases the sensitivity !!! metal detector example!
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0] # 0.5 is used as default
y_pred_prob[0:10]
y_pred_class[0:10] # predicts yes more often

confusion # the old confusion matrix
new_confusion = metrics.confusion_matrix(y_test, y_pred_class)
# sensitivity increases
# specificity has decreased
# both have an inverse relationship

# =============================================================================
# ROC and AUC
# =============================================================================

# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

# YOU CANNPT SEE WHICH THRESHOLD CORRESPONDS TO EACH PAIR OF sensitivity and specificity

# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    
evaluate_threshold(0.5)
evaluate_threshold(0.3)

# AUC is the percentage of the ROC plot that is underneath the curve:
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob))

# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

# =============================================================================
# Confusion metrix:
# Allows to calculate a variety of metrics
# Useful for multiclass problems
# ROC/AUC advantages:
# Does not require you to set a classification threshold
# Still useful when there is a high class imbalance
# =============================================================================

# =============================================================================
# How do I encode categorical feature using scikit-learn
# Improve accuracy
# with a pipeline
# =============================================================================
import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain')
df.shape
df.columns
df.isna().sum()
df = df.loc[df.Embarked.notna(), ['Survived', 'Sex', 'Pclass', 'Embarked']]
df.shape # we lost two columns who values of embarked were NaN
df.isna().sum() # Check data set has no Nan values
X = df.loc[:, ['Pclass']]
y = df.Survived

logreg = LogisticRegression(solver = 'lbfgs')
cross_val_score(logreg, X, y, cv = 5, scoring = 'accuracy').mean() # gives 0.6783
y.value_counts(normalize = True) # To give Null accuracy - 0.617

# Lets encode festures
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
ohe.fit_transform(df[['Sex']])
ohe.categories_
ohe.fit_transform(df[['Embarked']])

# Lets create a pipeline
X = df.drop('Survived', axis = 1)
from sklearn.compose import make_column_transformer
column_trans = make_column_transformer(
    (OneHotEncoder(), ['Sex', 'Embarked']),
    remainder = 'passthrough')
column_trans.fit_transform(X)

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, logreg)

cross_val_score(pipe, X, y, cv = 5, scoring = 'accuracy').mean()
X_new = X.sample(5, random_state = 99)
X_new
pipe.fit(X, y)
pipe.predict(X_new)

