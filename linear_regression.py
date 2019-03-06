# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:10:26 2019

@author: Munmun
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Beton.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y ,test_size=1/3, random_state=0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set result
y_pred= regressor.predict(X_test)

#visualizing the training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()