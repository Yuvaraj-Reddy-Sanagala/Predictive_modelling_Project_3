# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:16:06 2024

@author: Admin
"""

import pandas as pd

df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged.csv')

print()
print(df.dtypes)
print()
print(df.isnull().sum())