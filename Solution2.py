# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 23:35:45 2022

@author: darvi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import statsmodels.api as sm

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt
#%%
def read_file_get_points(filename):
    data = pd.read_json(filename, typ='series')    
    np_points = np.empty((0,2))
    for j in list(range(0, data.size)):
        if list(data[j].keys())[0] == 'points':            
            try:
                points = list(data[j].items())[0][1]
                for p in points:
                    ind = p[0]['location']
                    latitude = ind[0][0]['latitude']
                    longitude = ind[0][1]['longitude']
                    np_points = np.append(np_points, np.array([[latitude,longitude]]), axis=0)
            except:
                print('No detailed data recorded')
    return np_points
    
        
# Load JSON data files from data folder @author: h17163
folder = 'WorkoutData_2017to2020'
file_list = os.listdir(folder)

power = 10**4
# Read files to a common dataframe @author: h17163
for filename in file_list:
    print('\n'+ filename)
    np_points = read_file_get_points(folder + '/' + filename)
    np_points_filename = f'data/points/{filename}_points.csv'
    try:
        scaler = MinMaxScaler()
        scaler.fit(np_points)
        np_points = scaler.transform(np_points)
        np_points = np_points * power
        # np_points = np_points.astype(int)     
        np_points = np.int_(np_points)
    except:
        pass
    np.savetxt(np_points_filename, np_points, delimiter=',')
#%%
np_points_1 = genfromtxt("data/points/2017-01-01 08_54_23.0.json_points.csv", delimiter=',', dtype="int")
# np_points = np.int_(np_points_1)
np_mat = np.zeros((power+1,power+1))
np_mat = np_mat.astype(int)
for i in range(0, np_points.shape[0]):
    np_mat[np_points[i, 0], np_points[i, 1]] = 1

np_points_filename = 'testtest_points.csv'
np.savetxt(np_points_filename, np_mat, delimiter=',')
    