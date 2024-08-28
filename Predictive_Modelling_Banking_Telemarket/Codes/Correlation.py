# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:57:13 2024

@author: Admin
"""

import pandas as pd

df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_log_transform.csv')

num_col_names=df.iloc[:,7:].columns

df_corr_log = df[num_col_names].corr()['deposit']

