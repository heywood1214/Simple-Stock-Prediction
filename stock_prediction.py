#this programs predicts stock prices by using ML models

#install packages
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#get stock data
df = quandl.get("WIKI/VSP")
#Print data
print(df)

#get the adjusted close column
df = df[['Adj. Close']]
print(df.head())

#A variable for predicting 'n' days in the future, currently predicting 1 day in the future
forecast_out = 30

#create another column , shift one day 
df['Prediction']=df[['Adj. Close']].shift(-forecast_out)
#print the new data set
print(df.tail())

#create the indepedent data set 
# convert the dataframe into a numpy array
X = np.array(df.drop(['Prediction'],1))

#remove the last 'n' rows, for every column, we are going to get the last 30 rows 
X = X[:-forecast_out]
print(X)

##create dependent data set - Y
#convert datafram to a numpy array 
y = np.array(df['Prediction'])

#get all of the y values except the last 'n' rows
y = y[:-forecast_out]

#spliting data 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#SVM, radio basis kernel
svr_rbf = SVR(kernel ='rbf',C=1e3, gamma = 0.1)
svr_rbf.fit(x_train,y_train)

#testing model: score returns the coefficient of determination R^2 of the prediction 
# best score = 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print(svm_confidence)

#create and train the linear regression model 
linear_regression = LinearRegression()

#Train the model 
linear_regression.fit(x_train,y_train)
lr_confidence = linear_regression.score(x_test,y_test)
print(lr_confidence)

#set x_forecast equal to the last 30 rows of the original data set from adjusted closed
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

#print linear regressions predictions for the next 'n' days 
linear_regression_predict = linear_regression.predict(x_forecast)
print(linear_regression_predict)

#print support vector model predictions for the next 'n' days
svm_predict = svr_rbf.predict(x_forecast)
print(svm_predict)

