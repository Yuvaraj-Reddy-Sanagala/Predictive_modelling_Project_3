# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:31:12 2024

@author: Admin
"""
import pandas as pd
from feature_engine.encoding import OrdinalEncoder

df=pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_log_transform.csv')


cato_enc=OrdinalEncoder(encoding_method='arbitrary')

cato_enc_fit=cato_enc.fit(df)

cato_dict=cato_enc_fit.encoder_dict_
print()
print("Encoded Variables:")
for i in cato_dict:
    print()
    print(i+":",cato_dict[i])

cato_enc_transform=cato_enc_fit.transform(df)

cato_enc_transform.to_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_cato_encoded.csv',index=False)