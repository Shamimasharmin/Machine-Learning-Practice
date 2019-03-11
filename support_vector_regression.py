# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:46:45 2019

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

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#fitting support vector regression to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#predicting new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#y_pred = regressor.predict(6.5)

#visualizing SVR result
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff (Support Vector Regression)')
plt.xlabel('Level Position')
plt.ylabel('Salary')
plt.show()