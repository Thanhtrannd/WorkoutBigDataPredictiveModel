# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:48:04 2022

@author: darvi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from calendar import day_name
# from collections import deque

from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from mpl_toolkits.basemap import Basemap


import seaborn as sns

#%% Load dataset
df = pd.read_csv("data/df_res_full.csv")


#%% Pie chart function 

# Sport_calories_kcal = df.groupby("sport")["calories_kcal"].mean()
# Sport_calories_kcal = Sport_calories_kcal.to_frame()
# Sport_calories_kcal = Sport_calories_kcal.reset_index()

# def pie_chart(data, label):
    
#     # Wedge properties
#     wp = { 'linewidth' : 1, 'edgecolor' : "black" }
    
#     # Creating autocpt arguments
#     def make_autopct(values):
#         def my_autopct(pct):
#             total = sum(values)
#             val = int(round(pct*total/100.0))
#             return '{p:.1f}%\n({v:1d})'.format(p=pct,v=val)
#         return my_autopct
    
#     # Creating plot
#     fig, ax = plt.subplots(figsize =(30, 24))
#     wedges, texts, autotexts = ax.pie(data,
#                                       labels = label,
#                                       startangle = 90,
#                                       shadow = False,
#                                       wedgeprops = wp,
#                                       autopct = make_autopct(data))
    
#     # Adding legend
#     ax.legend(wedges, label,
#               title = "Sport:",
#               loc = "upper right")    
    
#     plt.setp(autotexts, size = 16, weight ="bold")
#     ax.set_title("Avg. Calories Burned (Kcal) w.r.t Sport", fontsize=26)
    
#     # Show plot
#     plt.show()


# pie_chart(Sport_calories_kcal["calories_kcal"], Sport_calories_kcal["sport"])

#%% Bar plot - WeekDay vs Sport

# Sport_duration_s = df.groupby(["sport","start_time_weekday_cat"])["duration_s"].mean()
# Sport_duration_s = Sport_duration_s.to_frame()
# Sport_duration_s = Sport_duration_s.reset_index()


# cats = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
# sport_unique = Sport_duration_s["sport"].unique()

# dict = {}
# for i in sport_unique:
#     dict[i] = Sport_duration_s[Sport_duration_s["sport"] == i]
#     dict[i] = dict[i].set_index('start_time_weekday_cat').reindex(cats).reset_index()
#     dict[i] = dict[i].fillna(0)
#     dict[i] = dict[i].to_numpy()
    


# plt.subplots(figsize =(20, 23))

# for j in sport_unique:
#     if j == sport_unique[0]:
#         plt.bar(dict[j][:,0].tolist(),dict[j][:,2])
#         bottom_ = dict[j][:,2]
#     else:    
#         plt.bar(dict[j][:,0],dict[j][:,2], bottom = bottom_)
#         bottom_ = bottom_ + dict[j][:,2]
#         print(bottom_)

 
    
# # Adding legend
# plt.xlabel("Weekday", fontsize = 23)
# plt.ylabel("Sport",fontsize = 23)
# plt.legend(sport_unique,
#           title = "Sport:",
#           loc = "upper center",fontsize = 10)
# plt.title("WeekDay vs Sport",fontsize = 30)
# plt.show()

#%% Bar plot - Seasons w.r.t sports

# Sport_distance_km = df.groupby(["sport","start_time_season_cat"])["distance_km"].mean()
# Sport_distance_km = Sport_distance_km.to_frame()
# Sport_distance_km = Sport_distance_km.reset_index()


# cats = ["AUTUMN", "SPRING", "SUMMER", "WINTER" ]
# sport_unique = Sport_distance_km["sport"].unique()

# dict = {}
# for i in sport_unique:
#     dict[i] = Sport_distance_km[Sport_distance_km["sport"] == i]
#     dict[i] = dict[i].set_index('start_time_season_cat').reindex(cats).reset_index()
#     dict[i] = dict[i].fillna(0)
#     dict[i] = dict[i].to_numpy()
    


# plt.subplots(figsize =(20, 23))

# for j in sport_unique:
#     if j == sport_unique[0]:
#         plt.bar(dict[j][:,0].tolist(),dict[j][:,2])
#         bottom_ = dict[j][:,2]
#     else:    
#         plt.bar(dict[j][:,0],dict[j][:,2], bottom = bottom_)
#         bottom_ = bottom_ + dict[j][:,2]
#         print(bottom_)

 
    
# # Adding legend
# plt.xlabel("Seasons", fontsize = 23)
# plt.ylabel("Avs.Distance Km",fontsize = 23)
# plt.legend(sport_unique,
#           title = "Sport:",
#           loc = "upper right",fontsize = 15)
# plt.title("Seasons w.r.t sports",fontsize = 30)
# plt.show()


#%% Multivariate Plots

# sns.set_style("ticks")
# sns.set(font_scale=1.3)
# plt.figure(figsize=(30,5))

# Sport_duration_s_1 = df.groupby(["sport","start_time_hour"])["duration_s"].mean()
# Sport_duration_s_1 = Sport_duration_s_1.to_frame()
# Sport_duration_s_1 = Sport_duration_s_1.reset_index()

# cats = list(range(2,23))
# sport_unique = Sport_duration_s_1["sport"].unique()

# dict = {}
# for i in sport_unique:
#     dict[i] = Sport_duration_s_1[Sport_duration_s_1["sport"] == i]
#     dict[i] = dict[i].set_index('start_time_hour').reindex(cats).reset_index()
#     dict[i] = dict[i].fillna(0)
#     dict[i] = dict[i].to_numpy()

# # Final solution
# cat = sns.catplot(
#     x="start_time_hour", 
#     y="duration_s", 
#     data=Sport_duration_s_1, 
#     height=5,
#     aspect=.8,
#     kind='point',
#     hue='sport', 
#     col='sport', 
#     col_wrap=6);

# cat.fig.subplots_adjust(top=.9)

# cat.fig.suptitle("Time vs Sport (24 hr)")

# for ax in cat.fig.axes:
#     ax.set_xlim(0,22)
#     ax.set_xticks(range(2,22,2))
#     ax.xaxis.tick_bottom()
#     ax.grid(True, axis='both')

# cat.set(xlabel="Hourly", ylabel = "Avg of Duration")


# # Alternative Solution 
# # rel = sns.relplot(x="start_time_hour",
# #                   y="duration_s", 
# #                   data=Sport_duration_s_1, 
# #                   height=5, #default 
# #                   aspect=.8,
# #                   palette='bright',
# #                   kind='line',
# #                   hue='sport', 
# #                   col='sport',
# #                   col_wrap=3)

# # g = sns.FacetGrid(Sport_duration_s_1, col="sport", row="sport")
# # g.map_dataframe(sns.histplot, x="start_time_hour", binwidth=2, binrange=(0, 60))

# # rel.fig.subplots_adjust(top=.95)

# # rel.fig.suptitle("Time vs Sport (24 hr)")

# # for ax in rel.fig.axes:
# #     ax.set_xlim(2,22)
# #     ax.set_xticks(range(2,22,2))

# # rel.set(xlabel="Hourly", ylabel = "Avg of Duration")
