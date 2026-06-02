# 📞 Telecom Customer Retention & Customer Risk Hub

## 📌 Executive Summary
Customer retention is a vital driver of profitability in the subscription and telecom space. This project develops an end-to-end machine learning pipeline that flags high-risk subscribers *before* they cancel their service. 

Using an optimized **XGBoost Classifier**, the system achieves a **high predictive accuracy (~95%)**, helping retention teams proactively intervene with targeted marketing strategies.

## ⚙️ Core Technical Pipeline
1. **Data Processing:** Consolidated usage data, handling categorical fields with `Scikit-Learn`'s `LabelEncoder` and numeric values with `StandardScaler`.
2. **Machine Learning Inference:** Implemented an optimized `XGBoost` model to handle tabular customer data configurations.
3. **Interactive User Interface:** Created an operational dashboard layout via `Streamlit` allowing customer success teams to run real-time predictive risk scenarios.

## 📈 Identified Business Insights
* **Customer Service Call Sensitivity:** Accounts logging more than 3 customer service interactions experience an exponential increase in churn probability.
* **International Plan Inflation:** Customers holding International Plans without a corresponding high volume of international calling minutes yield severe margin attrition.
