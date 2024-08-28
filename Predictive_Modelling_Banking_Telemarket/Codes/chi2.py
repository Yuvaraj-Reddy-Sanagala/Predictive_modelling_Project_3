# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:38:35 2024

@author: Admin
"""

import pandas as pd
from sklearn.feature_selection import chi2


df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_cato_encoded.csv')

X=df.iloc[:,0:7]
y=df.iloc[:,15]
f_score=chi2(X,y)
print(f_score)
pvalues=pd.Series(f_score[1])
print(pvalues)
pvalues.index=X.columns
pvalues_dict=dict(pvalues)

for i in pvalues_dict:
    if pvalues_dict[i]>0.05:
        df.drop([i],axis=1,inplace=True)

df.to_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_dataset.csv',index=False)

        
