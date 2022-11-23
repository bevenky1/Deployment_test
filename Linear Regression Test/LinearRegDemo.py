# imports

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

# Supress warnings
import warnings
warnings.filterwarnings('ignore')

# import dataset
advertising = pd.read_csv("advertising.csv")

# Visualization
# sns.pairplot(advertising, x_vars = ['TV', 'Radio', 'Newspaper'], y_vars = 'Sales', size = 5, kind = 'scatter')
# plt.show()

# sns.heatmap(advertising.corr(), cmap = "YlGnBu", annot = True)
# plt.show()

X = advertising["TV"]
Y = advertising["Sales"]

# Splitting the data into Train and Test (split ratio 70:30)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 5)

# adding constant (to get intercept)
x_train_sm = sm.add_constant(x_train)

# Fitting the regression model using OLS (of SM)
lr = sm.OLS(y_train, x_train_sm).fit()

# print(lr.summary())

# Error analysis
y_train_pred = lr.predict(x_train_sm)
res = y_train - y_train_pred


# For Test data   

# Adding constant
x_test_sm = sm.add_constant(x_test)

# predicting OLS model theat was built with train data
y_test_pred = lr.predict(x_test_sm)

res_test = y_test - y_test_pred

# Regression line for test (Assignment)
lr_test = sm.OLS(y_test, x_test_sm).fit()
print()
print("Parameters Statistics of the best fit line")
print(lr_test.params)

# Performance analysis
from sklearn.metrics import r2_score
r_sq = r2_score(y_test, y_test_pred)
print('R2 score  for the test data => ',r_sq)

from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_test_pred)
print('MAE score  for the test data => ', mae)

mse = metrics.mean_squared_error(y_test, y_test_pred)
print ('MSE score  for the test data => ',mse)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print ('RMSE score  for the test data => ', rmse)