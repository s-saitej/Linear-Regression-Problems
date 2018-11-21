# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:52:02 2018

@author: sunka
"""

import pandas as pd

dataframe = pd.read_csv('concrete.csv')

print(dataframe)


df1 = dataframe.iloc[:,1:]
df2 = dataframe.iloc[:,:8]


from sklearn import preprocessing

df1 = preprocessing.normalize(df1)

df2 = preprocessing.normalize(df2) 


from sklearn import model_selection

train_data , test_data , train_target , test_target = model_selection.train_test_split(df1,df2)



from sklearn import linear_model

regression = linear_model.LinearRegression()

fitting = regression.fit(train_data,train_target)

result = regression.predict(test_data)

print(result)


coefficient = regression.coef_
intercept = regression.intercept_

print("The coefficeint is " + str(coefficient))
print("Intercept is " + str(intercept))




from sklearn import metrics

mean_square_error = metrics.mean_squared_error(test_target,result)

print("Mean square error is " + str(mean_square_error))

varience = metrics.r2_score(test_target,result)

print("Varience is " + str(varience))




from matplotlib import pyplot

pyplot.hist(train_data)
pyplot.hist(result)


# Output

pyplot.scatter(test_target,result)
pyplot.title("Output")

pyplot.scatter(result,result-test_target)
pyplot.title('Residue')




