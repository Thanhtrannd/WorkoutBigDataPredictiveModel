# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 23:35:45 2022

@author: darvi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt
#%%
# def read_file_get_points(filename):
#     data = pd.read_json(filename, typ='series')    
#     np_points = np.empty((0,2))
#     for j in list(range(0, data.size)):
#         if list(data[j].keys())[0] == 'points':            
#             try:
#                 points = list(data[j].items())[0][1]
#                 for p in points:
#                     ind = p[0]['location']
#                     latitude = ind[0][0]['latitude']
#                     longitude = ind[0][1]['longitude']
#                     np_points = np.append(np_points, np.array([[latitude,longitude]]), axis=0)
#             except:
#                 pass
#     return np_points
    
        
# # Load JSON data files from data folder @author: h17163
# folder = 'WorkoutData_2017to2020'
# file_list = os.listdir(folder)

# # Read files to a common dataframe @author: h17163
# for filename in file_list:
#     filename_ = filename
#     filename_ = filename_.split('.')[0]
#     np_points = read_file_get_points(folder + '/' + filename)
#     np_points_filename = f'data/points/original/{filename_}.csv'
#     np.savetxt(np_points_filename, np_points, delimiter=',')


original_points_folder = 'data/points/original'
original_points_file_list = os.listdir(original_points_folder)

power = 10**4
for filename in original_points_file_list:
    np_points = np.genfromtxt(f"{original_points_folder}/{filename}", delimiter=',', dtype="int")
    try:
        scaler = MinMaxScaler()
        scaler.fit(np_points)
        np_points = scaler.transform(np_points)
        np_points = np_points * power    
        np_points = np.int_(np_points)
    except:
        pass

    np_mat = np.zeros((power+1,power+1))
    np_mat = np_mat.astype(int)
    for i in range(0, np_points.shape[0]):
        np_mat[np_points[i, 0], np_points[i, 1]] = 1

    filename_ = filename
    filename_ = filename_.split('.')[0]
    transformed_points_filename = f'data/points/transformed/{filename_}.csv'

    # Dataset is too large to be store to the repository. Therefore, in case, you want to test it, just uncomment the following line
    np.savetxt(transformed_points_filename, np_mat, delimiter=',')

    