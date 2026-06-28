# 🏦 Banking Fraud Detection using Machine Learning

## 📌 Overview

This project builds an end-to-end Machine Learning pipeline to detect fraudulent banking transactions using a real-world banking dataset. It covers data preprocessing, exploratory data analysis (EDA), feature engineering, feature selection, and classification using Logistic Regression and Decision Tree models.

---

## 🎯 Objective

Predict whether a banking transaction is **Fraudulent (1)** or **Legitimate (0)** using customer, account, and transaction-related features.

---

## 📂 Dataset

* **Rows:** 550,000
* **Columns:** 20 (Original)
* **Target Variable:** `is_fraud`

Features include:

* Transaction Amount
* Account Balance
* Credit Score
* Transaction Type
* Merchant Category
* Channel (UPI, NEFT, RTGS, IMPS)
* Account Type
* Loan Information
* KYC Status
* Transaction Date & Time
* State

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Statsmodels
* ScorecardPy

---

## 📊 Workflow

```text
Data Loading
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
VIF Analysis
      ↓
Weight of Evidence (WoE)
      ↓
Information Value (IV)
      ↓
Train-Test Split
      ↓
Logistic Regression
      ↓
Decision Tree
      ↓
Model Evaluation
```

## ⚙️ Feature Engineering

Created new features including:

* Transaction Year
* Transaction Month
* Transaction Day
* Time of Day (Morning / Afternoon / Evening / Night)

---

## 📈 Exploratory Data Analysis

Performed:

* Fraud vs Non-Fraud Distribution
* Transaction Amount Analysis
* Account Balance Analysis
* Credit Score Distribution
* Bivariate & Multivariate Analysis
* Outlier Detection using Boxplots

---

## 📉 Feature Selection

Performed:

* Variance Inflation Factor (VIF)
* Weight of Evidence (WoE)
* Information Value (IV)

---

## 🤖 Machine Learning Models

### Logistic Regression

* Binary Classification
* Model Evaluation using Accuracy, Precision, Recall and F1-Score

### Decision Tree Classifier

* Hyperparameter Tuning using GridSearchCV
* Decision Tree Visualization
* Performance Evaluation

---

## 📊 Results

* Logistic Regression achieved high overall accuracy but struggled to detect fraudulent transactions due to severe class imbalance.
* Decision Tree improved fraud detection (Recall) but produced more false positives.
* The dataset contains a highly imbalanced target distribution, making fraud prediction a challenging classification problem.


---

## 📚 Key Concepts

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* One-Hot Encoding
* VIF
* Weight of Evidence (WoE)
* Information Value (IV)
* Logistic Regression
* Decision Tree
* GridSearchCV
* Model Evaluation

---

## ⭐ Conclusion

This project demonstrates a complete machine learning workflow for fraud detection in banking transactions. It highlights the importance of data preprocessing, feature engineering, and model evaluation while addressing the challenges posed by highly imbalanced financial datasets.
