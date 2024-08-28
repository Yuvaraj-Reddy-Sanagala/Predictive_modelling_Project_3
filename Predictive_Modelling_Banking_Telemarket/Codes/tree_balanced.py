# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:28:28 2024

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
import ast
import graphviz


df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_balanced.csv')

f=open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Deci_Tree_balanced.txt","r")

content = f.readlines()

selected_features=content[6]

col_names = ast.literal_eval(selected_features)

f.close()


X=df[col_names]
y=np.ravel(df[['deposit']])

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.13,shuffle=True,random_state=42,stratify=y)

dtc=tree.DecisionTreeClassifier(max_depth=4)

dtc_fit=dtc.fit(X_train,y_train)

dot_data=tree.export_graphviz(dtc_fit, 
                               out_file=None,
                               feature_names=X_train.columns,
                               class_names=['Deposit','No_Deposit'],
                               filled=True,
                               rounded=True,
                               node_ids=True)

plt.show()

graph = graphviz.Source(dot_data)

graph.render("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Decision Tree Balanced Dataset")


feature_importance=dtc_fit.feature_importances_


dict_importance=dict()

for i in range(len(feature_importance)):
    key_name=col_names[i]
    dict_importance[key_name]=round(feature_importance[i],4)

print()
print("Dict of log_odds Balanced Dataset:\n",dict(sorted(dict_importance.items(), key=lambda item: item[1],reverse=True)))