# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:46:38 2024

@author: Admin
"""

import pandas as pd
import numpy as np

df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_imputed.csv')
col_names=['nr_employed','campaign','emp_var_rate','euribor3m','cons_price_idx']

for col in col_names:
    if min(df[col])<0:
        df[col+'_log']=np.log(df[col]+1-min(df[col]))
        df.drop([col],axis=1,inplace=True)
    else:
        df[col+'_log']=np.log(df[col]+1)
        df.drop([col],axis=1,inplace=True)

df_corr_log = df.iloc[:,10:].corr()['deposit']   
df.to_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_log_transform.csv',index=False)