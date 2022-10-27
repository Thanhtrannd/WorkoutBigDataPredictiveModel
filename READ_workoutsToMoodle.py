# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:06:17 2022

@author: h17163
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
#%% Settings
folder = 'C:\\Users\\darvi\\Desktop\\Big Data in Business and Industry\\PRACTICAL ASSIGNMENT-20221026\\WorkoutData_2017to2020'
file_list = os.listdir(folder)  

#%% Read example file
mov_ex0 = pd.read_json(folder+'/'+file_list[0], typ='series')

# Inspect example file
print(mov_ex0[14]['points'])

#%% Function definitions (HINT: you can move functions in a separate file to 
# keep the length of the analysis script reasonable...)

def read_file_to_df(filename):
    data = pd.read_json(filename, typ='series')
    value = []
    key = []
    for j in list(range(0, data.size)):
        if list(data[j].keys())[0] != 'points':
            key.append(list(data[j].keys())[0])
            value.append(list(data[j].items())[0][1])
            dictionary = dict(zip(key, value))
       

    if list(data[j].keys())[0] == 'points':
        try:
            start = list(list(list(data[data.size-1].items()))[0][1][0][0].items())[0][1][0]
            dictionary['start_lat'] = list(start[0].items())[0][1]
            dictionary['start_long'] = list(start[1].items())[0][1]
            dictionary['end_lat'] = list(start[0].items())[0][1]
            dictionary['end_long'] = list(start[1].items())[0][1]
        except:
            print('No detailed data recorded')
            
        
    df = pd.DataFrame(dictionary, index = [0])

    return df

#%% Read all files in a loop

# Create Empty DataFrame
df_res = pd.DataFrame()

# Read files to a common dataframe
for filename in file_list:
    print('\n'+filename)
    df_process = read_file_to_df(folder +'/'+ filename)
    df_res = pd.concat([df_res, df_process], 0)

df_res.reset_index(drop=True, inplace = True)

#%% Checking Nan Values

NaN_check = {}
for i in df_res.columns:
    check = df_res[i].isnull().values.any()
    if check == True:
        NaN_check[i] = df_res[i].isnull().sum()
    else:
        NaN_check[i] = 0



#%% START HERE

#%% DATA EXPLORATION

#%% Observe the number of NaN values in df_res

# no operations on var with many NaN, fill 1 existing NaN in var calories
# Fill 1 existing NaN in var calories
# observe which sport corresponds the NaN
# Compute calory burn rate of that particular sport = calories / duration
# Compute mean calory burn rate and replace NaN

calories_burned_rate = df_res["calories_kcal"] / df_res["duration_s"]
NaN_index = df_res.loc[pd.isna(df_res["calories_kcal"]), :].index
print(f"The index of NAN value in column 'alories_kcal' is {NaN_index[0]}.")
df_res.loc[NaN_index[0],"calories_kcal"] = np.mean(calories_burned_rate) * df_res.loc[NaN_index[0],"duration_s"]


#%% DATA PRETREATMENT

# Create dummy variables from categorial variable (source, sport)
sport_dummy = pd.get_dummies(df_res["sport"],prefix="sport")
source_dummy = pd.get_dummies(df_res["source"],prefix="source")

# Convert datetime variables to int
df_res["created_date_num"] = pd.to_datetime(df_res["created_date"])
df_res["created_date_num"] = df_res["created_date_num"].apply(lambda x: x.strftime("%Y%m%d%H%M%S"))
df_res["created_date_num"] = df_res["created_date_num"].apply(lambda x: float(x))

df_res["start_timee_num"] = pd.to_datetime(df_res["start_time"])
df_res["start_timee_num"] = df_res["start_timee_num"].apply(lambda x: x.strftime("%Y%m%d%H%M%S"))
df_res["start_timee_num"] = df_res["start_timee_num"].apply(lambda x: float(x))

df_res["end_time_num"] = pd.to_datetime(df_res["end_time"])
df_res["end_time_num"] = df_res["end_time_num"].apply(lambda x: x.strftime("%Y%m%d%H%M%S"))
df_res["end_time_num"] = df_res["end_time_num"].apply(lambda x: float(x))



#%% SOLUTION 1:
# Create dataset for solution 1 => DF1 (9 columns)
# Contains variables without NaN
NAN_check_1 = NaN_check
NAN_check_1["calories_kcal"] = 0
column_remove = []
for i in NAN_check_1:
    if NAN_check_1[i] != 0:
        column_remove.append(i)

# Remove columns including NaN value more than 1
DF1 = df_res[df_res.columns[~df_res.columns.isin(column_remove)]]

# DF1_class
DF1_sport = DF1["sport"]
nObs = sport_dummy.shape[0]
# DF1_class = sport_dummy.iloc[7:]
DF1_class = DF1_sport.iloc[7:]
DF1_class = DF1_class.to_numpy()



del DF1["created_date"]
del DF1["start_time"]
del DF1["end_time"]
del DF1["sport"]
del DF1["source"]


DF1 = pd.concat([DF1, source_dummy, sport_dummy], axis = 1)

# Min-Max normalization
DF1["duration_s"] = (DF1["duration_s"]-DF1["duration_s"].min())/(DF1["duration_s"].max()-DF1["duration_s"].min())
DF1["distance_km"] = (DF1["distance_km"]-DF1["distance_km"].min())/(DF1["distance_km"].max()-DF1["distance_km"].min())
DF1["calories_kcal"] = (DF1["calories_kcal"]-DF1["calories_kcal"].min())/(DF1["calories_kcal"].max()-DF1["calories_kcal"].min())
DF1["speed_avg_kmh"] = (DF1["speed_avg_kmh"]-DF1["speed_avg_kmh"].min())/(DF1["speed_avg_kmh"].max()-DF1["speed_avg_kmh"].min())
DF1["created_date_num"] = (DF1["created_date_num"]-DF1["created_date_num"].min())/(DF1["created_date_num"].max()-DF1["created_date_num"].min())
DF1["start_timee_num"] = (DF1["start_timee_num"]-DF1["start_timee_num"].min())/(DF1["start_timee_num"].max()-DF1["start_timee_num"].min())
DF1["end_time_num"] = (DF1["end_time_num"]-DF1["end_time_num"].min())/(DF1["end_time_num"].max()-DF1["end_time_num"].min())

DF1_data = pd.DataFrame()
# Create shifted dataset including past events as additional columns
for j in DF1.columns:
    for i in range(1,8): 
        name = j+"_"+str(i)
        DF1_data[name] = DF1[j].shift(i)
        
DF1_data = DF1_data.iloc[7:,:]
DF1_data = DF1_data.to_numpy()

#%% Construct model for next SPORT prediction (KNN, CLUSTERING K-MEANS, GAUSSIAN MODELS) using DF1_1
X_train, X_test, y_train, y_test = train_test_split(DF1_data, DF1_class, random_state=42, test_size=0.3)
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_5 = knn_model.predict(X_test)
print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
print("Hello")


# Construct model for next TIME prediction (LINEAR REGRESSION) using DF1_1

# Construct model for next DURATION prediction (LINEAR REGRESSION) using DF1_1