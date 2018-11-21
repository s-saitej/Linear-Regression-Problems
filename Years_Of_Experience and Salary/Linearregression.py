# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:43:47 2018

@author: sunkara
"""


# Reading Data from dataset

import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
print(dataset)



# Splitting data from dataset as independant and dependant variables.
# In the above data set years of experience is independant and salary is dependant on years of exp.
# Here we get y=mx+c where y is dependant on x.
# y = Salary and x = Years of Experience


x = dataset.iloc[:,:1]
y = dataset.iloc[:,1:]

#After splitting the data we need to get testing data to verify the solution and trainging data to train our model.
# Training data will be present on x-axis and training target will be on y-axis.
# Testing data will be present on x-axis and testing-target will be on y-axis.

from sklearn import model_selection

train_data, test_data ,train_target ,test_target  = model_selection.train_test_split(x,y)

# Now after having x and y values we need to fit the trainging data (x) and training target (y) in the graph.
# Since data is continuous we will be using linear regression, as salary increases with years of experience.

from sklearn import linear_model

regression = linear_model.LinearRegression() # We are isntructing pc to apply linear regression

fiting = regression.fit(train_data,train_target) # We are fixing train data on x and train target on y axis


# Now we have fitted the train_data and train_target on x and y axis.

# To look at the fitted curve we plot a graph

from matplotlib import pyplot

pyplot.scatter(train_data,train_target)

# Now it was to find result of un-known values to model, i.e testing data.
# To predict the values we use,

result = regression.predict(test_data)


# Here the main motive of regression if to find the slope and intercept of a regresion line. Eventhough result is already predicted we need to observe intercept and slope.

slope = regression.coef_
intercept  = regression.intercept_

# Now to validate whether the predicted value is true or false we will be comparing result with testing target. Because testing target will contain actual values and result contain predicted values

pyplot.scatter(result,test_target)
pyplot.hist(result, color='orange')
pyplot.hist(test_target, color='blue')

# Now to find the residual of the result,

pyplot.scatter(result,result-test_target)


# Here an error can be found through using mean_square_error and accuracy can be calculated by using varience.
# An error will be present between result and testing target. So we require both result and testing target to find error
# Varience is the deviation of result from the actual testing target. If deviation is less then varience will be more.
# Thus varience can be used to test the accuracy of a model.

from sklearn import metrics

mean_square_error = metrics.mean_squared_error(result,test_target)
varience = metrics.r2_score(result,test_target)