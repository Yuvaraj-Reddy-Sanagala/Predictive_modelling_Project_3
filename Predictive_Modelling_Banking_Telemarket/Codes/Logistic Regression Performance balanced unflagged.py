# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:57:11 2024

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import ast


df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_balanced.csv')

f=open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Log_Reg_balanced_unflagged.txt","r")

content = f.readlines()

selected_features=content[8]

col_names = ast.literal_eval(selected_features)

f.close()


X=df[col_names]
y=np.ravel(df[['deposit']])


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.13,shuffle=True,random_state=42,stratify=y)

lr=LogisticRegression(max_iter=10000)

lr_fit=lr.fit(X_train,y_train)

y_predict=lr_fit.predict(X_test)

confusion_matrix=metrics.confusion_matrix(y_test,y_predict)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Not_Deposit","Deposit"])


cm_display.plot(cmap='Blues',colorbar=False) 
plt.ylabel('Actual')
plt.xlabel('Prediction')
plt.title('Logistic_Regression_Balanced_Unflagged')


Accuracy = round(metrics.accuracy_score(y_test, y_predict),5)

Precision = round(metrics.precision_score(y_test, y_predict),5)

Precision_1 = round(metrics.precision_score(y_test, y_predict,pos_label=0),5)

Sensitivity = round(metrics.recall_score(y_test, y_predict),5)

Specificity = round(metrics.recall_score(y_test, y_predict,pos_label=0),5)

F1_score = round(metrics.f1_score(y_test, y_predict),5)

Fbeta_score=round(metrics.fbeta_score(y_test, y_predict,beta=0.5),5)

print()
print("Logistic Regression Balanced Unflagged:")
print()
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity":Sensitivity,"Specificity":Specificity,"F1_score":F1_score, "Fbeta_score":Fbeta_score})



