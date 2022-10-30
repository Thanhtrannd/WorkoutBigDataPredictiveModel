# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:48:04 2022

@author: darvi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Pie chart function 

df = pd.read_csv("data/df_res_filled.csv")
Sport = df.groupby("sport")["calories_kcal"].mean()
Sport = Sport.to_frame()
Sport = Sport.reset_index()

def pie_chart(data, label):
    
    # Wedge properties
    wp = { 'linewidth' : 1, 'edgecolor' : "black" }
    
    # Creating autocpt arguments
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.1f}%\n({v:1d})'.format(p=pct,v=val)
        return my_autopct
    
    # Creating plot
    fig, ax = plt.subplots(figsize =(30, 24))
    wedges, texts, autotexts = ax.pie(data,
                                      labels = label,
                                      startangle = 90,
                                      shadow = False,
                                      wedgeprops = wp,
                                      autopct = make_autopct(data))
    
    # Adding legend
    ax.legend(wedges, label,
              title = "Sport:",
              loc = "upper right")    
    
    plt.setp(autotexts, size = 16, weight ="bold")
    ax.set_title("Avg. Calories Burned (Kcal) w.r.t Sport", fontsize=26)
    
    # Show plot
    plt.show()


pie_chart(Sport["calories_kcal"], Sport["sport"])

#%% Pie chart function 

def weekDay_stacked_bar_chart(data, label):
    if len(label) > 1:     
        plt.bar(label, data1)
        plt.bar(label, data2, bottom=data1)
        plt.bar(label, data3, bottom=data1+data2)
        
        # Adding legend
        plt.xlabel("Weekday")
        plt.ylabel("Sport")
        plt.legend(label,
                  title = "Sport:",
                  loc = "upper right")
        plt.title("WeekDay vs Sport")
        
        # Show plot
        plt.show()

