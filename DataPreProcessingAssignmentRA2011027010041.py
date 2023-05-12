# -*- coding: utf-8 -*-
"""Data Preprocessing Assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XHcSiErUKLq9SJGh7ebL-67TO_pIs3tE
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

cols_to_normalize = ['Adult_Mortality','Infant_Deaths','Hepatitis ',	'Measles ',' BMI ','Underfive_Deaths ',	'Polio','Diphtheria ',' HIV','GDP','Population','Malnourished10_19 ','Malnourished5_9','Income_Index','Schooling']

dataset[cols_to_normalize] = scaler.fit_transform(dataset[cols_to_normalize])

dataset.head()