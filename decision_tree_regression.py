# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:49:09 2019

@author: Munmun
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, [2]].values

#fitting decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#predicting new result
y_pred = regressor.predict(6.5)

#visualizing decision tree regression result
X_grid = np.arange( min(X), max(X), 0.01)
X_grid=X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or bluff (Decision Tree Regression)')
plt.xlabel('Level Position')
plt.ylabel('Salary')
plt.show()