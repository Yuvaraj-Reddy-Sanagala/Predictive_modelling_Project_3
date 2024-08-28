# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:24:46 2024

@author: Admin
"""

import pandas as pd
from feature_engine.imputation import CategoricalImputer

df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_flag.csv')

cato_imputer=CategoricalImputer(imputation_method='frequent')

imputed_df=cato_imputer.fit_transform(df)

imputed_df.to_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_imputed.csv',index=False)