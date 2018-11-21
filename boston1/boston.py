import sklearn

data = sklearn.datasets.load_boston()

from sklearn import preprocessing

data.data = preprocessing.normalize(data.data)


from sklearn.linear_model import LinearRegression

regression = LinearRegression()
fitting = regression.fit(data.data,data.target)

result = regression.predict(data.data)

coefficient = regression.coef_
intercept = regression.intercept_

print(coefficient)
print(intercept)

from sklearn.metrics import mean_squared_error,r2_score

mean_square_error = mean_squared_error(data.target,result)
vairence = r2_score(data.target,result)

print(mean_square_error)
print(vairence)


from matplotlib import pyplot

pyplot.hist(data.data )
pyplot.hist(result, color = 'orange')
pyplot.hist(data.target, color = 'blue')

pyplot.scatter(result,data.target , color = 'orange')
pyplot.title("Result")
