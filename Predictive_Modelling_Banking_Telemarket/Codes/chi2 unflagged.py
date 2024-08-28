# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:58:02 2024

@author: Admin
"""

import pandas as pd
from sklearn.feature_selection import chi2


df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_cato_encoded.csv')

X=df.iloc[:,0:10]
y=df[['deposit']]

f_score=chi2(X,y)
print(f_score)
pvalues_unflagged=pd.Series(f_score[1])
print(pvalues_unflagged)
pvalues_unflagged.index=X.columns
pvalues_dict_unflagged=dict(pvalues_unflagged)

for i in pvalues_dict_unflagged:
    if pvalues_dict_unflagged[i]>0.05:
        df.drop([i],axis=1,inplace=True)

df.to_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_dataset.csv',index=False)