# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:08:35 2022

@author: darvi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from datetime import datetime



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn import linear_model

df = pd.read_csv("data/df_res_filled.csv")

#%% Checking Nan Values


NaN_check = {}
for i in df.columns:
    check = df[i].isnull().values.any()
    if check == True:
        NaN_check[i] = df[i].isnull().sum()
    else:
        NaN_check[i] = 0
    
#%% Converting datetime variable "created_date" to float

df["created_date_num"] = pd.to_datetime(df["created_date"])
df["created_date_num"] = df["created_date_num"].apply(lambda x: x.strftime("%Y%m%d%H%M%S"))
df["created_date_num"] = df["created_date_num"].apply(lambda x: float(x))


#%% Converting datetime variable "start_time" and "end_time" to categories

# Converting datetime variable "start_time" to categories
df["start_time"] = pd.to_datetime(df["start_time"])
df["start_time"] = df["start_time"].apply(lambda x: x.strftime("%H%M%S"))
df["start_time"] = df["start_time"].apply(lambda x: int(x))


# Converting datetime variable "end_time" to categories
df["end_time"] = pd.to_datetime(df["end_time"])
df["end_time"] = df["end_time"].apply(lambda x: x.strftime("%H%M%S"))
df["end_time"] = df["end_time"].apply(lambda x: int(x))


# Start transforming
morning_default_left = 50000
morning_default_right = 115959


afternoon_default_left = 120000
afternoon_default_right = 165959

evening_default_left = 170000
evening_default_right = 205959

df["start_time_trans"] = df["start_time"]
df["end_time_trans"] = df["end_time"]



for i in range(0,len(df["start_time"])):
    if morning_default_right >= df.loc[i,"start_time"] >= morning_default_left:
        df.loc[i,"start_time_trans"] = "MORNING"
    elif afternoon_default_right >= df.loc[i,"start_time"] >= afternoon_default_left:
        df.loc[i,"start_time_trans"] = "AFTERNOON"
    elif evening_default_right >= df.loc[i,"start_time"] >= evening_default_left:
        df.loc[i,"start_time_trans"] = "EVENING"
    else:
        df.loc[i,"start_time_trans"] = "NIGHT"


for i in range(0,len(df["end_time"])):
    if morning_default_right >= df.loc[i,"end_time"] >= morning_default_left:
        df.loc[i,"end_time_trans"] = "MORNING"
    elif afternoon_default_right >= df.loc[i,"end_time"] >= afternoon_default_left:
        df.loc[i,"end_time_trans"] = "AFTERNOON"
    elif evening_default_right >= df.loc[i,"end_time"] >= evening_default_left:
        df.loc[i,"end_time_trans"] = "EVENING"
    else:
        df.loc[i,"end_time_trans"] = "NIGHT"

df = df.iloc[:,[-1,-2] + list(range(0,16))]
   
#%% One-hot encoding categorical/ string data variables

# Create dummy for "sport" variable
sport_dummy = pd.get_dummies(df["sport"],prefix="sport")


# Create dummy for "source" variable
source_dummy = pd.get_dummies(df["source"],prefix="source") 


# Create dummy for "start_time_trans" variable
start_time_trans_dummy = pd.get_dummies(df["start_time_trans"],prefix="start_time_trans") 


# Create dummy for "end_time_trans" variable
end_time_trans_dummy = pd.get_dummies(df["end_time_trans"],prefix="end_time_trans") 


#%% Merge df dtaframe with dummy dataframes
df1 = pd.concat([df, sport_dummy, source_dummy, start_time_trans_dummy, end_time_trans_dummy], axis = 1)


#%% Normalizing data
for i in range(7, 18):
    
    df1.iloc[:,i] = (df1.iloc[:,i]-df1.iloc[:,i].min())/(df1.iloc[:,i].max()-df1.iloc[:,i].min())

#%% shifting timeseries

lst = df1.columns.tolist()
for j in range(7, len(df1.columns)):
    for i in range(1,8): 
        name = lst[j]+"_"+str(i)
        df1[name] = df1[lst[j]].shift(i)


#%% Generate datasets for alternative 1

# df1_data_KNN
column_remove = df.columns.tolist()

df1_data_LR = pd.DataFrame()
df1_data_LR = df1.iloc[7:,:]
df1_data_LR = df1_data_LR[df1_data_LR.columns[~df1_data_LR.columns.isin(column_remove)]]
df1_data_LR = df1_data_LR.to_numpy()



# df1_class_KNN
df1_class_LR = df1["created_date_num"]
nObs = df1.shape[0]
df1_class_LR = df1_class_LR.iloc[7:]
df1_class_LR = df1_class_LR.apply(lambda x: float(x))
df1_class_LR = df1_class_LR.to_numpy()
df1_class_LR = df1_class_LR.astype(np.float)

#%% Construct model for next TIME prediction (Ridge) using df1
X_train, X_test, y_train, y_test = train_test_split(df1_data_LR, df1_class_LR, random_state=42, test_size=0.2)

model = Ridge(alpha = 1)
model.fit(X_train, y_train)

coefficient  = model.score(X_train, y_train)
print(f"coefficient of determination: {coefficient}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

y_pred = model.predict(X_test)
print(f"predicted response:\n{y_pred}")

accuracy = r2_score(y_test, y_pred)
print(f"Accuracy is {accuracy}")

mean_absolut_error = mean_absolute_error(y_test, y_pred)
print(f"mean_absolute_error is {mean_absolut_error}")

mean_square_error = mean_squared_error(y_test, y_pred)
print(f"mean_squared_error is {mean_square_error}")

pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
display(pred_df)

#%% Construct model for next TIME prediction (LINEAR REGRESSION) using DF1_1

# X_train, X_test, y_train, y_test = train_test_split(df1_data_LR, df1_class_LR, random_state=42, test_size=0.3)

# model = LinearRegression().fit(X_train, y_train)

# coefficient  = model.score(X_train, y_train)
# print(f"coefficient of determination: {coefficient}")

# print(f"intercept: {model.intercept_}")

# print(f"slope: {model.coef_}")

# y_pred = model.predict(X_test)
# print(f"predicted response:\n{y_pred}")

# accuracy = r2_score(y_test, y_pred)
# print(f"Accuracy is {accuracy}")

# mean_absolut_error = mean_absolute_error(y_test, y_pred)
# print(f"mean_absolute_error is {mean_absolut_error}")

# mean_square_error = mean_squared_error(y_test, y_pred)
# print(f"mean_squared_error is {mean_square_error}")

# pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# display(pred_df)






















