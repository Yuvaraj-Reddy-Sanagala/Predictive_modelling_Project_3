# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:40:02 2024

@author: Admin
"""

import pandas as pd

df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_flag.csv')

print()
print(df.dtypes)
print()
print(df.isnull().sum())