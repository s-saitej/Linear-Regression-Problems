# -*- coding: utf-8 -*-
"""
Created on Thu May 24 23:16:10 2018

@author: sunka
"""

import pandas as pd

dataframe = pd.read_csv('concrete.csv')

data = dataframe.iloc[:,:4]
target = dataframe.iloc[:,4:]





from sklearn import model_selection

data_train_1 , data_train_2 , target_1 , target_2 = model_selection.train_test_split(data,target) #One pair for training and other pair for testing





from sklearn import linear_model

regression = linear_model.LinearRegression()

fitting = regression.fit(data_train_1,target_1)

result = regression.predict(data_train_2)

coefficient = regression.coef_

intercept = regression.intercept_





from sklearn import metrics

mean_square_error = metrics.mean_squared_error(target_2,result)

print(mean_square_error)

varience = metrics.r2_score(target_2,result)

print(varience)





import matplotlib.pyplot as plt

plt.hist(data_train_1)
plt.hist(result)


plt.scatter(target_2,result)
plt.scatter(result,result-target_2)

