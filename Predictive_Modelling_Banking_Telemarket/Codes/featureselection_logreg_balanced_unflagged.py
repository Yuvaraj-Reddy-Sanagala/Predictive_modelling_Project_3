# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:47:42 2024

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector

df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_balanced.csv')

X = df.iloc[:, 0:16]
y=np.ravel(df[['deposit']])

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.13,shuffle=True,random_state=42,stratify=y)

sfs_logreg = SequentialFeatureSelector(linear_model.LogisticRegression(max_iter=10000),
                                k_features=9,
                                floating=True,
                                forward=True,
                                cv=5)
sfs_logreg.fit(X_train, y_train)

f=open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Log_Reg_balanced_unflagged.txt","w")

for i in sfs_logreg.subsets_:
    features_selected=str(list(sfs_logreg.subsets_[i]['feature_names']))
    f.write(features_selected+"\n")


f.close()