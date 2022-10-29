# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:31:07 2022

@author: darvi
"""

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay




df = pd.read_csv("data/df_res_filled.csv")

#%% Checking Nan Values


# NaN_check = {}
# for i in df.columns:
#     check = df[i].isnull().values.any()
#     if check == True:
#         NaN_check[i] = df[i].isnull().sum()
#     else:
#         NaN_check[i] = 0
    
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

df1_data_KNN = pd.DataFrame()
df1_data_KNN = df1.iloc[7:,:]
df1_data_KNN = df1_data_KNN[df1_data_KNN.columns[~df1_data_KNN.columns.isin(column_remove)]]
df1_data_KNN = df1_data_KNN.to_numpy()



# df1_class_KNN
df1_class_KNN = df1["sport"]
nObs = df1.shape[0]
df1_class_KNN = df1_class_KNN.iloc[7:]
df1_class_KNN = df1_class_KNN.apply(lambda x: str(x))
df1_class_KNN = df1_class_KNN.to_numpy()
df1_class_KNN = df1_class_KNN.astype(np.str)

#%% Construct model for next SPORT prediction (KNN, CLUSTERING K-MEANS, GAUSSIAN MODELS) using df1

X_train, X_test, y_train, y_test = train_test_split(df1_data_KNN, df1_class_KNN, random_state=42, test_size=0.2)
Acc_lst = {}
for i in range(3,20,2):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)*100
    Acc_lst[i] = acc

best_acc = max(Acc_lst)

s1_sport_model = KNeighborsClassifier(n_neighbors=11)
s1_sport_model.fit(X_train, y_train)
y_pred_11 = s1_sport_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_11)*100

print(f"Accuracy with k=9 is {accuracy}")

confusion_matrix = confusion_matrix(y_test, y_pred_11)
print(confusion_matrix)

names = df1["sport"].unique()
names = names.tolist()

cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = names)
cm_display.plot()
plt.show()

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred_11)))
print('Precision: {:.2f}'.format(precision_score(y_test, y_pred_11,average='weighted')))
print('Recall: {:.2f}'.format(recall_score(y_test, y_pred,average='weighted')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred,average='weighted')))



    




















