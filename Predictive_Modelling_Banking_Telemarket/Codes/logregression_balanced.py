# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:29:22 2024

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ast


df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_balanced.csv')

f=open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Log_Reg_balanced.txt","r")

content = f.readlines()

selected_features=content[6]

col_names = ast.literal_eval(selected_features)

f.close()


X=df[col_names]
y=np.ravel(df[['deposit']])


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.10,shuffle=True,random_state=42,stratify=y)

lr=LogisticRegression(max_iter=10000)

lr_fit=lr.fit(X_train,y_train)

log_odds,intercept=np.exp(lr_fit.coef_),lr_fit.intercept_
dict_log_odds=dict()

for i in range(len(log_odds[0])):
    key_name=col_names[i]
    dict_log_odds[key_name]=round(log_odds[0][i],4)
    
print()
print("Dict of log_odds Balanced Dataset:\n",dict(sorted(dict_log_odds.items(), key=lambda item: item[1],reverse=True)))

