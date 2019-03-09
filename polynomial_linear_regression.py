# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:55:20 2019

@author: Munmun
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualizing linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Level Position')
plt.ylabel('Salary')
plt.show()

#visualizing polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Level Position')
plt.ylabel('Salary')
plt.show()

#predicting a new result with linear regression
#lin_reg.predict(6.5)

#predicting new result with polynomial regression
#lin_reg_2.predict(poly_reg.fit_transform(6.5))




