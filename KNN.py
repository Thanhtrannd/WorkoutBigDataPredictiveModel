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

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


        
#%%
df = pd.read_csv("data/DF1_SPORT_X.csv")
df_1 = pd.read_csv("data/df_res_filled.csv")
X_train = pd.read_csv("data/DF1_SPORT_Xtrain.csv")
X_test = pd.read_csv("data/DF1_SPORT_Xtest.csv")
Y_train = pd.read_csv("data/DF1_SPORT_Ytrain.csv")
Y_test = pd.read_csv("data/DF1_SPORT_Ytest.csv")


#%% PCA
def plottingConfusionMatrix(model, Xtest, Ytest):
    matrix = plot_confusion_matrix(model, Xtest, Ytest, cmap=plt.cm.Blues)
    matrix.ax_.set_title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.gcf().axes[0].tick_params()
    plt.gcf().axes[1].tick_params()
    plt.show()
    
pca = PCA(n_components=110)
pca.fit(X_train)
A = pca.explained_variance_ratio_.cumsum()
print(pca.explained_variance_ratio_.cumsum())

lst = []
A_filter = A[(A>=0.6) & (A<=0.8)]
for i in A_filter:
    result = np.where(A == i)
    lst.append(result)
index_explained_variance_ratio_80 = lst[0][0][0]
index_explained_variance_ratio_85 = lst[-1][0][0]

for j in range(index_explained_variance_ratio_80, index_explained_variance_ratio_85+1):
    pca = PCA(n_components=j)
    PCA_train  = pca.fit_transform(X_train)
    PCA_test  = pca.transform(X_test)
    knn_model = KNeighborsClassifier(n_neighbors=15)
    knn_model.fit(PCA_train, Y_train)
    y_pred = knn_model.predict(PCA_test)
    acc = accuracy_score(Y_test, y_pred)*100
    plottingConfusionMatrix(knn_model, PCA_test, Y_test)

#%%

# Acc_lst = {}
# for i in range(3,20,2):
#     knn_model = KNeighborsClassifier(n_neighbors=i)
#     knn_model.fit(X_train, Y_train)
#     y_pred = knn_model.predict(X_test)
#     acc = accuracy_score(Y_test, y_pred)*100
#     Acc_lst[i] = acc

# best_acc = max(Acc_lst)

# s1_sport_model = KNeighborsClassifier(n_neighbors=11)
# s1_sport_model.fit(X_train, Y_train)
# y_pred_11 = s1_sport_model.predict(X_test)
# accuracy = accuracy_score(Y_test, y_pred_11)*100
# print(f"Accuracy with k=11 is {accuracy}")


#%% DecisionTreeClassifier
# clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
# clf_gini.fit(X_train, Y_train)


# clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
# clf_entropy.fit(X_train, Y_train)

# y_pred_gini = clf_gini.predict(X_test)

# y_pred_entropy = clf_entropy.predict(X_test)

# print(classification_report(Y_test, y_pred_gini))

# print(classification_report(Y_test, y_pred_entropy))

# plt.figure(figsize = (25,10))
# a  = tree.plot_tree(clf_gini, filled = True, rounded = True, fontsize = 14)

# b  = tree.plot_tree(clf_entropy, filled = True, rounded = True, fontsize = 14)

    

# A = df.groupby(df["sport"].tolist(),as_index=False).size()



















