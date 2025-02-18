#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


df = pd.read_csv(r'/content/customer_churn.csv')
df


# Data Inspection

# In[34]:


df.head()


# In[35]:


df.tail()


# In[36]:


df.shape


# In[37]:


df.columns


# In[38]:


df.info()


# In[39]:


df.describe()


# In[40]:


df.isnull().sum()


# # Data Cleaning
# 

# In[42]:


df.duplicated().sum()


# In[43]:


for i in df.columns:
  print(df[i].unique())


# In[44]:


df.nunique() 


# In[45]:


list = ['a','b','@','c']


# In[46]:


df['TotalCharges'] = pd.to_numeric(df["TotalCharges"],errors = 'coerce')


# In[47]:


df['TotalCharges']


# In[20]:


df.info()


# In[21]:


df


# In[25]:


df.sample(n=100) 


# In[27]:


df.iloc[::3]


# In[31]:


df.groupby('gender')['MonthlyCharges'].sum()


# In[30]:


df.groupby('gender').sample(n=10)


# In[50]:


df.isnull().sum()


# In[51]:


df.dropna(inplace = True)


# In[52]:


for col in df.columns:
  if df[col].dtype != 'object':
    plt.boxplot(df[col])
    plt.title(col)
    plt.show()


# In[54]:


df.drop(columns =['customerID'],inplace = True)


# In[55]:


df


# In[56]:


from sklearn.preprocessing import LabelEncoder


# In[57]:


le = LabelEncoder()


# In[58]:


for col in df.columns:
  if df[col].dtype == 'object':
    df[col] = le.fit_transform(df[col])


# In[59]:


df


# In[60]:


df.info()


# In[61]:


df.corr()


# In[64]:


plt.figure(figsize = (20,10))
sns.heatmap(df.corr(),annot = True,cmap = 'Oranges')


# Model Building

# In[65]:


x = df.iloc[:,:-1]
y = df['Churn']


# In[66]:


x


# In[67]:


y


# In[68]:


from sklearn.model_selection import train_test_split


# In[135]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,random_state = 45)


# In[136]:


x_train


# In[137]:


y_train


# In[138]:


from sklearn.linear_model import LogisticRegression


# In[139]:


LR = LogisticRegression()


# In[140]:


LR.fit(x_train,y_train)


# In[141]:


LRPred = LR.predict(x_test)


# In[142]:


LRPred


# In[143]:


from sklearn.metrics import *


# In[144]:


accuracy_score(y_test,LRPred)


# In[145]:


from sklearn.tree import DecisionTreeClassifier


# In[163]:


DT = DecisionTreeClassifier(max_depth = 8)


# In[ ]:


# DT.fit(x_train,y_train)


# In[165]:


DTPred = DT.predict(x_test)


# In[166]:


accuracy_score(y_test,DTPred)


# In[150]:


from sklearn.ensemble import RandomForestClassifier


# In[167]:


RF = RandomForestClassifier(n_estimators = 100, criterion = 'entropy' , random_state = 0)


# In[168]:


RF.fit(x_train,y_train)


# In[169]:


RFPred = RF.predict(x_test)


# In[170]:


accuracy_score(y_test,RFPred)


# In[ ]:




