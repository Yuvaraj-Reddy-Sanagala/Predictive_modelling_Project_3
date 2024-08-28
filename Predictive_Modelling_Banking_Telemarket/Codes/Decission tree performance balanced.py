
"""
Created on Sun Mar 31 21:30:34 2024

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import ast


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

y_predict=dtc_fit.predict(X_test)

confusion_matrix=metrics.confusion_matrix(y_test,y_predict)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Not_Deposit","Deposit"])

cm_display.plot(cmap='Blues',colorbar=False)
plt.title('Decision_Tree_Balanced_Flagged')
plt.ylabel('Actual')
plt.xlabel('Prediction')

Accuracy = round(metrics.accuracy_score(y_test, y_predict),5)

Precision = round(metrics.precision_score(y_test, y_predict),5)

Sensitivity = round(metrics.recall_score(y_test, y_predict),5)

Specificity = round(metrics.recall_score(y_test, y_predict,pos_label=0),5)

F1_score = round(metrics.f1_score(y_test, y_predict),5)

Fbeta_score=round(metrics.fbeta_score(y_test, y_predict,beta=0.5),5)


print()
print("Decision Tree Balanced:")
print()
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity":Sensitivity,"Specificity":Specificity,"F1_score":F1_score, "Fbeta_score":Fbeta_score})

