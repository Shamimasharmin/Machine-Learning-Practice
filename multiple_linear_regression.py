# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:14:08 2019

@author: Munmun
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing datasets
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features= [3])
X=onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X=X[:, 1:]

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y ,test_size=0.2, random_state=0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set result
y_pred= regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)





















