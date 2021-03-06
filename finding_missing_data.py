# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing datasets
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
#from sklearn.impute import SimpleImputer(for new version)
imputer= Imputer(missing_values='NaN',strategy='mean', axis=0)
imputer=imputer.fit(X[: ,1:3])
X[: ,1:3]=imputer.transform(X[: ,1:3])