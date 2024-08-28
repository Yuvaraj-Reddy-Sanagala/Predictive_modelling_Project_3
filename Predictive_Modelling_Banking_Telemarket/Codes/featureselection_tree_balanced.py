# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:07:43 2024

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector

df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_balanced.csv')

X = df.iloc[:, 0:12]
y=np.ravel(df[['deposit']])

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.13,shuffle=True,random_state=42,stratify=y)

sfs_tree = SequentialFeatureSelector(tree.DecisionTreeClassifier(max_depth=4),
                                k_features=7,
                                floating=True,
                                forward=True,
                                cv=2)
sfs_tree.fit(X_train, y_train)

f=open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Deci_tree_balanced.txt","w")

for i in sfs_tree.subsets_:
    features_selected=str(list(sfs_tree.subsets_[i]['feature_names']))
    f.write(features_selected+"\n")

f.close()