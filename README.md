# Customer-Churn
Creating a classification models that will help classify whether or not a customer churned.

📊 Customer Churn Prediction
This project focuses on building a classification model to predict whether a customer will churn. Customer churn, or attrition, refers to when a customer stops using a company's product or service. By identifying at-risk customers, businesses can take proactive measures to retain them.

🔍 Project Overview
Objective: Develop a machine learning model to classify customers as either churned or retained.
Dataset: Contains customer behavior, demographic details, and subscription information.
Models Used: Logistic Regression
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.

📂 Key Files
customer_churn.csv/ - Contains the dataset (Link given).
Customer_Churn.py/ - Exploratory Data Analysis (EDA) and model development.
Accuracy Score- Trained models and evaluation results.
Scripts for data preprocessing, feature engineering, and model training.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'/content/customer_churn.csv')
df
df.head()

![image](https://github.com/user-attachments/assets/4708f08c-3369-469d-96b3-3f8d04269628)

df.columns

![image](https://github.com/user-attachments/assets/2580b0e9-c833-4912-8086-a786eeb64199)

![image](https://github.com/user-attachments/assets/8dba23bb-54fb-42cc-9f65-e9b946f7befa)

![image](https://github.com/user-attachments/assets/c41d5f7b-ee63-4272-b649-a318e334d3ed)

![image](https://github.com/user-attachments/assets/ab2104e9-0011-406e-9264-51eb52ac61f5)

![image](https://github.com/user-attachments/assets/b71d27c7-9ca8-44ab-9457-eff570333b86)

![image](https://github.com/user-attachments/assets/57b620b1-8b68-41b8-ae47-ed83c9ce8bd1)

![image](https://github.com/user-attachments/assets/f6983e2d-265d-4f3e-b511-5b7b2d5bd540)

![image](https://github.com/user-attachments/assets/2415cc81-ba0e-44fc-a941-74925b85cd31)

df.drop(columns =['customerID'],inplace = True)

![image](https://github.com/user-attachments/assets/9b14a04e-5eef-4203-a372-979ef3586a83)

![image](https://github.com/user-attachments/assets/06c76fec-ab47-4ba2-9852-e448d947d4cf)

![image](https://github.com/user-attachments/assets/fb15d230-c94d-4ccf-b169-22e20880bdac)

![image](https://github.com/user-attachments/assets/9c0c1e8c-a7ec-4d37-8807-681bacc84a52)

Model Building

![image](https://github.com/user-attachments/assets/c927dd91-668e-460a-8900-1609776d0312)

![image](https://github.com/user-attachments/assets/1831eab0-6a4e-4d39-aad6-5bb3c6915574)

Logistic Regression

![image](https://github.com/user-attachments/assets/b8d0d16e-5b3a-4a1b-98aa-64d78cb868e0)

![image](https://github.com/user-attachments/assets/84eb5815-7c4e-42df-8de7-908e011d2890)

![image](https://github.com/user-attachments/assets/66889870-3640-4c0f-9757-473e6c3152ac)





















