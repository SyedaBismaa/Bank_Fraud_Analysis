# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:53:49 2026

@author: syeda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load DAta 
df=pd.read_csv(r"C:\Users\syeda\OneDrive\Desktop\Banking_Fraud\indian_banking_transactions.csv")

#data Overview 
print(df.head(10))
print(df.shape)
print(df.isna().sum())
print(df.columns)

#Drop Useless cols


#Bi and multi variant analysis 

pd.crosstab(df["credit_score"], df["is_fraud"]).plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("credit score vs Fraud")
plt.xlabel("credit score")
plt.ylabel("Number of Transactions")
plt.legend(["Not Fraud", "Fraud"])
plt.show()

pd.crosstab(
    [df["account_type"], df["channel"]],
    df["is_fraud"]
).plot(
    kind="bar",
    stacked=True,
    figsize=(12,6)
)

plt.title("acc type & channel vs Fraud")
plt.xlabel("acc type and channel")
plt.ylabel("frouds")
plt.legend(["Not Fraud", "Fraud"])
plt.show()

#One hot encoding
df=pd.get_dummies(df,columns=["account_type"],dtype=int,drop_first=True)
df=pd.get_dummies(df,columns=["transaction_type"],dtype=int, drop_first=True)
df=pd.get_dummies(df,columns=["transaction_direction"], dtype=int, drop_first=True)
df=pd.get_dummies(df,columns=["transaction_status"], dtype=int, drop_first=True)
df=pd.get_dummies(df,columns=["kyc_status"], dtype=int, drop_first=True)
df=pd.get_dummies(df,columns=["channel"], dtype=int, drop_first=True)  
df =pd.get_dummies(df,columns=["state"], dtype=int, drop_first=True)
df=pd.get_dummies(df,columns=["merchant_category"], dtype=int, drop_first=True)


#Null vals
df["loan_type"]=df["loan_type"].fillna("No Loan")   #= have null vals 

df=pd.get_dummies(df,columns=["loan_type"],dtype=int, drop_first=True)


#Feature eng 
df["transaction_date"]=pd.to_datetime(df['transaction_date'], dayfirst=True)
df["transaction_year"]=df["transaction_date"].dt.year
df["transaction_month"]= df["transaction_date"].dt.month
df["transaction_day"]=df["transaction_date"].dt.day
#dropTransaction_date (already created features)
df=df.drop("transaction_date",axis=1)

#morning_afternoon-evening-Night_
def time_category(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"

df["Time_of_Day"] = df["transaction_hour"].apply(time_category)
df=pd.get_dummies(df,columns=["Time_of_Day"],dtype=int,drop_first=True)
print(df.columns)

df.to_csv(
    r"C:\Users\syeda\OneDrive\Desktop\Banking_Fraud\Clean_banking_data.csv",
    index=False
)

print("Saved Sucessfully !")