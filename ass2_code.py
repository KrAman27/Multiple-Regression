# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 07:27:20 2020

@author: aman kumar
"""
"""Air Quality Prediction using Multiple Regression"""

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the datasets
train_set = pd.read_csv('Train (1).csv')
X_test = pd.read_csv('Test.csv')
X_train = train_set.iloc[:,:-1].values
y_train = train_set.iloc[:,5] 

#fitting Multiple regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#building the optimal model using Backward elimination
#import statsmodel.regression.linear_model as sm

#Cheking the score  
print('Train Score: ', regressor.score(X_train, y_train))  
#print('Test Score: ', regressor.score(X_test, y_pred))
        
#building the optimal model using backward elimination
import statsmodels.regression.linear_model as sm
X_train = np.append(arr=np.ones((1600,1)).astype(int),values = X_train,axis=1)

X_opt_train=  X_train[:,[0,1,2,3,4,5]]
regressor_OLS_train=sm.OLS(endog=y_train,exog=X_opt_train).fit()
regressor_OLS_train.summary()

#building the optimal model using backward elimination
import statsmodels.regression.linear_model as sm
X_test = np.append(arr=np.ones((400,1)).astype(int),values = X_test,axis=1)

X_opt_test = X_test[:,[0,1,2,3,4,5]]
regressor_OLS_test = sm.OLS(endog=y_pred,exog=X_opt_test).fit()
regressor_OLS_test.summary()
