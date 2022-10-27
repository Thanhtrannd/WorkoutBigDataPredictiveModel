# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:06:17 2022

@author: h17163
"""

import pandas as pd
import os

#%% Settings
folder = './WorkoutData_2017to2020'
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
    count = 0
    for j in list(range(0, data.size)):
        if list(data[j].keys())[0] != 'points':
            key.append(list(data[j].keys())[0])
            value.append(list(data[j].items())[0][1])
            dictionary = dict(zip(key, value))
       

    if list(data[j].keys())[0] == 'points':
        
        count = len(data[j]['points'])
        try:
            start = list(list(list(data[data.size-1].items()))[0][1][0][0].items())[0][1][0]
            dictionary['start_lat'] = list(start[0].items())[0][1]
            dictionary['start_long'] = list(start[1].items())[0][1]
            dictionary['end_lat'] = list(start[0].items())[0][1]
            dictionary['end_long'] = list(start[1].items())[0][1]
        except:
            print('No detailed data recorded')
            
        
    df = pd.DataFrame(dictionary, index = [0])

    return df, count

#%% Read all files in a loop

# Create Empty DataFrame
df_res = pd.DataFrame()

count = list()
# Read files to a common dataframe
for filename in file_list:
    print('\n'+filename)
    df_process, pointCount = read_file_to_df(folder +'/'+ filename)
    df_res = pd.concat([df_res, df_process], 0)
    count.append(pointCount)
    
df_res.reset_index(drop=True, inplace = True)

#%%
# FROM HERE
# Solution 1: Let use a compact df with only variables having no NaN

cols = df_res.columns
nCol = len(cols)
nNaN = [0] * nCol

for i in range(nCol):
    nNaN[i] = df_res[cols[i]].isnull().sum()



