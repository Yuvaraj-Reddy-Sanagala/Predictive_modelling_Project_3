# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:51:44 2024

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import ast


df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_unflagged_balanced.csv')

f_DTC_unflagged = open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Deci_Tree_balanced_unflagged.txt", "r")

content_DTC_unflagged = f_DTC_unflagged.readlines()

selected_features_DTC_unflagged = content_DTC_unflagged[8]

col_names_DTC_unflagged = ast.literal_eval(selected_features_DTC_unflagged)

f_DTC_unflagged.close()


X_DTC_unflagged = df[col_names_DTC_unflagged]
y_DTC_unflagged = np.ravel(df[['deposit']])


X_train_DTC_unflagged, X_test_DTC_unflagged, y_train_DTC_unflagged, y_test_DTC_unflagged = train_test_split(X_DTC_unflagged,y_DTC_unflagged, test_size=0.48, shuffle=True, random_state=42, stratify=y_DTC_unflagged)

dtc = tree.DecisionTreeClassifier(max_depth=4)

dtc_fit = dtc.fit(X_train_DTC_unflagged, y_train_DTC_unflagged)

y_proba_DTC_unflagged = dtc_fit.predict_proba(X_test_DTC_unflagged)[:,1]

auc_score_DTC_unflagged = round(metrics.roc_auc_score(y_test_DTC_unflagged, y_proba_DTC_unflagged), 5)

fpr_DTC_unflagged, tpr_DTC_unflagged, threshold_DTC_unflagged = metrics.roc_curve(y_test_DTC_unflagged, y_proba_DTC_unflagged)



f_lr_unflagged=open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Log_Reg_balanced_unflagged.txt","r")

content_lr_unflagged = f_lr_unflagged.readlines()

selected_features_lr_unflagged=content_lr_unflagged[8]

col_names_lr_unflagged = ast.literal_eval(selected_features_lr_unflagged)

f_lr_unflagged.close()


X_lr_unflagged=df[col_names_lr_unflagged]
y_lr_unflagged=np.ravel(df[['deposit']])


X_train_lr_unflagged, X_test_lr_unflagged, y_train_lr_unflagged, y_test_lr_unflagged=train_test_split(X_lr_unflagged, y_lr_unflagged, test_size=0.13,shuffle=True,random_state=42,stratify=y_lr_unflagged)

lr_unflagged=LogisticRegression(max_iter=10000)

lr_unflagged_fit=lr_unflagged.fit(X_train_lr_unflagged,y_train_lr_unflagged)

y_proba_lr_unflagged = lr_unflagged_fit.predict_proba(X_test_lr_unflagged)[:,1]

auc_score_lr_unflagged = round(metrics.roc_auc_score(y_test_lr_unflagged, y_proba_lr_unflagged), 5)

fpr_lr_unflagged, tpr_lr_unflagged, threshold_lr_unflagged = metrics.roc_curve(y_test_lr_unflagged, y_proba_lr_unflagged)


df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Bank_Term_Deposit_balanced.csv')

f_DTC_flagged = open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Deci_Tree_balanced.txt","r")

content_DTC_flagged = f_DTC_flagged.readlines()

selected_features_DTC_flagged = content_DTC_flagged[6]

col_names_DTC_flagged = ast.literal_eval(selected_features_DTC_flagged)

f_DTC_flagged.close()


X_DTC_flagged = df[col_names_DTC_flagged]
y_DTC_flagged = np.ravel(df[['deposit']])


X_train_DTC_flagged, X_test_DTC_flagged, y_train_DTC_flagged, y_test_DTC_flagged = train_test_split(X_DTC_flagged,y_DTC_flagged, test_size=0.13, shuffle=True, random_state=42, stratify=y_DTC_flagged)

dtc = tree.DecisionTreeClassifier(max_depth=4)

dtc_fit = dtc.fit(X_train_DTC_flagged, y_train_DTC_flagged)

y_proba_DTC_flagged = dtc_fit.predict_proba(X_test_DTC_flagged)[:,1]

auc_score_DTC_flagged = round(metrics.roc_auc_score(y_test_DTC_flagged, y_proba_DTC_flagged), 5)

fpr_DTC_flagged, tpr_DTC_flagged, threshold_DTC_flagged = metrics.roc_curve(y_test_DTC_flagged, y_proba_DTC_flagged)



f_lr_flagged=open("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\Features_Selected_Log_Reg_balanced.txt","r")

content_lr_flagged = f_lr_flagged.readlines()

selected_features_lr_flagged=content_lr_flagged[6]

col_names_lr_flagged = ast.literal_eval(selected_features_lr_flagged)

f_lr_flagged.close()


X_lr_flagged=df[col_names_lr_flagged]
y_lr_flagged=np.ravel(df[['deposit']])


X_train_lr_flagged, X_test_lr_flagged, y_train_lr_flagged, y_test_lr_flagged=train_test_split(X_lr_flagged, y_lr_flagged, test_size=0.10,shuffle=True,random_state=42,stratify=y_lr_flagged)

lr_flagged=LogisticRegression(max_iter=10000)

lr_flagged_fit=lr_flagged.fit(X_train_lr_flagged,y_train_lr_flagged)

y_proba_lr_flagged = lr_flagged_fit.predict_proba(X_test_lr_flagged)[:,1]

auc_score_lr_flagged = round(metrics.roc_auc_score(y_test_lr_flagged, y_proba_lr_flagged), 5)

fpr_lr_flagged, tpr_lr_flagged, threshold_lr_flagged = metrics.roc_curve(y_test_lr_flagged, y_proba_lr_flagged)

plt.title("ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.plot(fpr_DTC_unflagged,tpr_DTC_unflagged,ls='--')
plt.plot(fpr_DTC_flagged,tpr_DTC_flagged,ls='--')
plt.plot(fpr_lr_unflagged,tpr_lr_unflagged,ls='--')
plt.plot(fpr_lr_flagged,tpr_lr_flagged,ls='--')
plt.plot([0,1],[0,1],ls="--",c="black")
plt.legend(["Decion Tree Unflagged AUC: "+str(auc_score_DTC_unflagged),"Decion Tree AUC: "+str(auc_score_DTC_flagged),"Logistic Regression Unflagged AUC: "+str(auc_score_lr_unflagged),"Logistic Regression AUC: "+str(auc_score_lr_flagged)],prop = { "size": 8 },loc="lower right")
plt.savefig("C:\\Users\\Admin\\Desktop\\Subjects\\BUS 602 Strategy for Business Analytics\\Presnetation\\data transformations\\ROC",dpi=175)
