# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:05:30 2026

@author: syeda
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scorecardpy as sc

#Models 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
df2=pd.read_csv(r"C:\Users\syeda\OneDrive\Desktop\Banking_Fraud\Clean_banking_data.csv")

print(df2.dtypes)

#drop_objects 
df2=df2.drop(["transaction_id", "customer_id","transaction_time"], axis=1)

print(df2.dtypes)

X = df2.drop(columns=["is_fraud"])

X_sample = X.sample(n=50000, random_state=42)

vif=pd.DataFrame()

vif["Feature"]=X.columns
vif["VIF"] = [
    variance_inflation_factor(X_sample.values, i)
    for i in range(X_sample.shape[1])
]

print(vif)

#based of vif dropping useless cols
df2=df2.drop(columns=["transaction_year" ,"transaction_hour"],axis=1)


#woe
df_woe=df2.drop(columns=["is_fraud"])
# Generate WoE bins
bins = sc.woebin(df2, y="is_fraud")
# Print WoE table for one feature
print(bins["credit_score"])

#iv
iv = sc.iv(df2, y="is_fraud")
print(iv)

#EMI Amount and Transaction Amount exhibited the highest IV(Sus)

#Models -?

#1- LinearRegression 
y=df2["is_fraud"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
    )

model=LogisticRegression(
    random_state=42,
    max_iter=1000
    )
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Decision Tree
model=DecisionTreeClassifier()

param_grid={
    "max_depth":[3,5,7,10,15],
    "min_samples_split":[2,5,10],
    "criterion":["gini","entropy"]
    }

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1"
    )

grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))