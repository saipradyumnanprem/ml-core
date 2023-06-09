# -*- coding: utf-8 -*-
"""Data_For_Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eRnL06Qv3ZFl1wIwjYGbSGIf7ZGFl622
"""

import numpy as np
import pandas as pd

dataset= pd.read_csv("Life_Expectancy.csv")

dataset.head()

dataset.shape

dataset.info

dataset.isnull().sum()

dataset.dropna(axis=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Define the columns to normalize
cols_to_normalize = ['Adult_Mortality','Infant_Deaths','Hepatitis ',	'Measles ',' BMI ','Underfive_Deaths ',	'Polio','Diphtheria ',' HIV','GDP','Population','Malnourished10_19 ','Malnourished5_9','Income_Index','Schooling']

# Normalize the selected columns
dataset[cols_to_normalize] = scaler.fit_transform(dataset[cols_to_normalize])

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X=dataset.drop('Country',axis=1).drop('Status',axis=1).drop('Expected', axis=1)
Y=dataset['Expected']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=50)

Reg=LinearRegression()

Reg.fit(X_train,Y_train)

Y_pred=Reg.predict(X_test)

from sklearn.metrics import r2_score

r2_score(Y_test, Y_pred)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(Y_test, Y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))