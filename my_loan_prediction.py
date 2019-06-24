#!/usr/bin/env python
# coding: utf-8

# # Importing packages

# In[1]:


# For working with data

import pandas as pd 

# For spliting the data before training
    
from sklearn.model_selection import train_test_split

# For using LogisticRegression algorithm
    
from sklearn.linear_model import LogisticRegression

# For checking the accuracy of train model
    
from sklearn.metrics import accuracy_score


# # Load Data

# In[2]:


train = pd.read_csv('train.csv') # train data
test = pd.read_csv('test.csv') # test data


# # Preview data

# In[3]:


train.head() # first 5 columns of the train data


# In[4]:


train.shape


# # Univariate Analysis

# In[5]:


train['ApplicantIncome'].hist()


# In[6]:


train['CoapplicantIncome'].hist()


# In[7]:


train['Loan_Amount_Term'].value_counts(normalize = True).plot.bar()


# In[8]:


train['Gender'].value_counts(normalize = True).plot.bar()


# In[9]:


train['Credit_History'].value_counts(normalize = True).plot.bar()


# In[10]:


train['Property_Area'].value_counts(normalize = True).plot.bar()


# In[11]:


train['Loan_Status'].value_counts(normalize = True).plot.bar()


# # Bi-variate Analysis

# In[12]:


pd.crosstab(train['Gender'],train['Loan_Status']).plot.bar(stacked = True)


# In[13]:


pd.crosstab(train['Education'],train['Loan_Status']).plot.bar(stacked = True)


# In[14]:


pd.crosstab(train['Self_Employed'],train['Loan_Status']).plot.bar(stacked = True)


# In[15]:


pd.crosstab(train['Dependents'],train['Loan_Status']).plot.bar(stacked = True)


# In[16]:


pd.crosstab(train['Married'],train['Loan_Status']).plot.bar(stacked = True)


# In[17]:


pd.crosstab(train['Property_Area'],train['Loan_Status']).plot.bar(stacked = True)


# In[18]:


pd.crosstab(train['Gender'],train['Credit_History']).plot.bar(stacked = True)


# In[19]:


pd.crosstab(train['Self_Employed'],train['Credit_History']).plot.bar(stacked = True)


# In[20]:


pd.crosstab(train['Dependents'],train['Credit_History']).plot.bar(stacked = True)


# In[21]:



pd.crosstab(train['Married'],train['Credit_History']).plot.bar(stacked = True)


# In[22]:


pd.crosstab(train['Property_Area'],train['Credit_History']).plot.bar(stacked = True)


# # Finding outliers

# In[23]:


import seaborn as sns
sns.boxplot(train['ApplicantIncome'])


# In[24]:


sns.boxplot(train['CoapplicantIncome'])


# In[25]:


sns.boxplot(train['LoanAmount'])


# # Dealing with outliers

# In[26]:


Q1 = train.quantile(0.25) # Quartile 1
Q3 = train.quantile(0.75) # Quartile 2


# In[27]:


IQR = Q3 - Q1  # Quartile Range


# In[28]:


IQR


# In[29]:


appl = train['ApplicantIncome']


# In[30]:


outliers = (train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))


# In[31]:


outliers['ApplicantIncome'].value_counts()


# In[32]:


outliers['LoanAmount'].value_counts()


# In[33]:


outliers['CoapplicantIncome'].value_counts()


# In[34]:


train_out = train[~((train < (Q1 - 1.5 * IQR)) |(train> (Q3 + 1.5 * IQR))).any(axis=1)]


# In[35]:


train_out['ApplicantIncome'].value_counts().sum()


# # Dealing with missing value

# In[36]:



train.isnull().sum()


# In[37]:


train['Gender'].fillna(train['Gender'].mode()[0],inplace = True)


# In[38]:


train['Gender'].isnull().sum()


# In[39]:


def Fill(key):
    k = str(key)
    train[k].fillna(train[k].mode()[0],inplace = True)


# In[40]:


Fill('Dependents')
Fill('Education')
Fill('Married')
Fill('Self_Employed')
Fill('Credit_History')
Fill('Married')


# In[41]:


def Fill_Continuous(key):
    k = str(key)
    train[k].fillna(train[k].mean(),inplace = True)


# In[42]:


Fill_Continuous('LoanAmount')
Fill_Continuous('Loan_Amount_Term')


# In[43]:



train.isnull().sum()


# In[44]:


train['Loan_Amount_Term'].value_counts()


# In[45]:


train['Gender'].replace({'Male':0, 'Female':1},inplace = True)

train['Education'].replace({'Graduate':1,'Not Graduate':0},inplace = True)

train['Self_Employed'].replace({'Yes':0,'No':1},inplace = True)

train['Loan_Status'].replace({'Y':0,'N':1},inplace = True)


train['income'] = train['ApplicantIncome'] + train['CoapplicantIncome']

train['Married'].replace({'Yes':0,'No':1},inplace = True)

train['Dependents'].replace({'3+':'3'},inplace= True)


# In[46]:


train.dtypes


# In[47]:


test['Gender'].fillna(test['Gender'].mode()[0],inplace = True)


# In[48]:


def Fill_Continuous(key):
    k = str(key)
    test[k].fillna(test[k].mean(),inplace = True)


# In[49]:


def Fill(key):
    k = str(key)
    test[k].fillna(test[k].mode()[0],inplace = True)


# In[50]:


Fill('Education')
Fill('Self_Employed')
Fill('Dependents')
Fill('Credit_History')


# In[51]:


Fill_Continuous('LoanAmount')

Fill_Continuous('Loan_Amount_Term')


# In[52]:



test.describe()


# In[53]:



test.isnull().sum()


# In[54]:


test['Education'].replace({'Graduate':1,'Not Graduate':0},inplace = True)

test['Married'].replace({'Yes':0,'No':1},inplace = True)


test['Self_Employed'].replace({'Yes':0,'No':1},inplace = True)

test['income'] = train['ApplicantIncome'] + train['CoapplicantIncome']

test['Gender'].replace({'Male':0, 'Female':1},inplace = True)

test['Dependents'].replace({'3+':'3'},inplace= True)




# In[55]:


train.isnull().sum()


# In[56]:


test.isnull().sum()


# In[57]:


train_data = train.drop('Loan_ID',axis = 1)
test_data = test.drop('Loan_ID',axis = 1)


# In[58]:


train_data.info()


# In[59]:


test_data.info()


# # creating model 

# In[60]:


#predictor  variable

X = train_data.drop('Loan_Status',1) 

#target variable

Y = train_data.Loan_Status 


# In[61]:


# creating dummies data

X = pd.get_dummies(X)
train_dummies = pd.get_dummies(train_data)
test_dummies = pd.get_dummies(test_data)


# In[62]:


train_data.shape


# In[63]:


test_data.shape


# In[64]:


train_dummies.shape


# In[65]:


test_dummies.shape


# In[66]:


#spliting data

x_train,x_cv,y_train,y_cv = train_test_split(X,Y,test_size = 0.3)


# In[67]:


#fiting data to model

model = LogisticRegression()
model.fit(x_train,y_train)


# In[68]:


#predicting data on test data

pred_cv = model.predict(x_cv) 


# In[69]:


#checking accuracy of model

accuracy_score(y_cv,pred_cv)


# In[70]:


#predicting data on test dummies data

pred_test = model.predict(test_dummies)


# # creating csv files

# In[71]:


#imporing sample submission csv

submission=pd.read_csv("sample_submission.csv")


# In[72]:


#putting values of predicted values on test dummies

submission['Loan_Status']=pred_test 


# In[73]:


#taking loan id from test data and saving 

submission['Loan_ID']=test['Loan_ID']


# In[74]:


#replacing 0 with N and 1 with Y

submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[75]:


#Exporting data to csv

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')

