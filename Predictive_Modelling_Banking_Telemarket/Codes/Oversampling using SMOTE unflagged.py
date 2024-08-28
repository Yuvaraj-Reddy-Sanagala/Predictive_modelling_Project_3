# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:07:08 2024

@author: Admin
"""

import pandas as pd
from imblearn.over_sampling import SMOTE


df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_dataset.csv')

X = df.iloc[:, 0:16]
y=df[['deposit']]


smote = SMOTE(random_state=42)

X_Resampled,y_Resampled = smote.fit_resample(X, y)

balanced_df=X_Resampled.join(y_Resampled)

balanced_df.to_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_balanced.csv',index=False)