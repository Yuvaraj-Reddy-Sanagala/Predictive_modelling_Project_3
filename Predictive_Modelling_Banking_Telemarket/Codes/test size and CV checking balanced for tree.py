
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:47:15 2024

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_balanced.csv')

X = df.iloc[:, 0:12]
y=np.ravel(df[['deposit']])


print()
print("Decision Tree for Balanced Dataset")
print()

for i in range(2,11,1):

    k_folds = KFold(n_splits = i, shuffle=True, random_state=42)
    k=dict()
        
    for j in range(10,51,1):
            
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=(j/100),shuffle=True,random_state=42,stratify=y)

        est = DecisionTreeClassifier(max_depth=4)
        
        scores = cross_val_score(est, X_train, y_train, cv = k_folds)
        
        k["test_size="+str(j)+"% CV="+str(i)]=round(scores.mean(),5)
        
    for i in k:
        if k[i]==max(k.values()):
            print(i,k[i])

