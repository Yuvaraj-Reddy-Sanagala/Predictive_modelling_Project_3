# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:40:20 2024

@author: Admin
"""

import pandas as pd
from feature_engine.imputation import CategoricalImputer

df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged.csv')

cato_imputer=CategoricalImputer(imputation_method='frequent')

imputed_df=cato_imputer.fit_transform(df)

imputed_df.to_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_imputed.csv',index=False)