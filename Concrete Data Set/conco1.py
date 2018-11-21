# -*- coding: utf-8 -*-
"""
Created on Thu May 24 23:47:52 2018

@author: sunka
"""

import pandas as pd

dataframe = pd.read_csv('concrete.csv')

print(dataframe)

df1 = dataframe.iloc[:,:8]
df2 = dataframe.iloc[:,2:]





from sklearn import model_selection

test_data , train_data , test_target , train_target = model_selection.train_test_split(df1,df2)





from sklearn import linear_model

regression = linear_model.LinearRegression()

curve_fitting = regression.fit(train_data,train_target)

result = regression.predict(test_data)





coefficient = regression.coef_
intercept = regression.intercept_




from sklearn import metrics

mean_square_error = metrics.mean_squared_error(test_target,result)
varience = metrics.r2_score(test_target,result)

print("error is "+str(mean_square_error))

print("Varience is "+str(varience))







import matplotlib.pyplot as plt

# output

plt.scatter(test_target,result)



#histogram output

plt.hist(test_target)
plt.hist(result)










