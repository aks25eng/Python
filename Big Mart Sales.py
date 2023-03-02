#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Problem Statement

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.

Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.


### Hypothesis Generation

Make it a practice to do this before solving any ML problem. 
Ideally,before seeing the data or else, you might end up with biased hypotheses.

What could affect the target variable (sales)?

1. Time of week : Weekends usually are more busy
2. Time of day  : Higher sales in the mornings and late evenings
3. Time of year : Higher sales at end of the year 
4. Store size and location
5. Items with more shelf space


# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


# In[2]:


train = pd.read_csv('bigmart_train.csv')


# In[4]:


train.head(20)


# In[5]:


train.shape


# In[6]:


train.isnull().sum()


# <font color=red|pink|yellow>**learner tasks (intentionally skipped)**</font>
# 
# **Exploratory Data Analysis**
# 
# 1. Univariate analysis on 
#  1. Target variable - Item outlet sales (histogram)
#  1. Independent variables (numeric and categorical) - histograms
# 
# 2. Bivariate analysis
#  1.  Explore IV's  with respect to the target variable - scatterplots
#  
# 3. Correlation matrix

# In[7]:


train['Item_Fat_Content'].unique()
#notice Low fat, Low Fat, LF are all the same variable


# In[8]:


train['Outlet_Establishment_Year'].unique()


# In[9]:


train['Outlet_Age'] = 2018 - train['Outlet_Establishment_Year']
train.head()


# In[10]:


train['Outlet_Size'].unique()


# In[11]:


train.describe()


# In[12]:


train['Item_Fat_Content'].value_counts()


# In[13]:


train['Outlet_Size'].value_counts()


# In[14]:


train['Outlet_Size'].mode()[0]


# In[15]:


# fill the na for outlet size with medium
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])


# In[16]:


# fill the na for item weight with the mean of weights
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].mean())


# In[17]:


train['Item_Visibility'].hist(bins=20)


# In[18]:


# delete the observations

Q1 = train['Item_Visibility'].quantile(0.25)
Q3 = train['Item_Visibility'].quantile(0.75)
IQR = Q3 - Q1
filt_train = train.query('(@Q1 - 1.5 * @IQR) <= Item_Visibility <= (@Q3 + 1.5 * @IQR)')


# In[19]:


filt_train


# In[20]:


filt_train.shape, train.shape


# In[21]:


train = filt_train
train.shape


# In[22]:


#train['Item_Visibility'].value_counts()


# In[22]:


#creating a category
train['Item_Visibility_bins'] = pd.cut(train['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])


# In[23]:


train['Item_Visibility_bins'].value_counts()


# In[24]:


train['Item_Visibility_bins'] = train['Item_Visibility_bins'].replace(np.nan,'Low Viz',regex=True)


# In[25]:


train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')


# In[26]:


train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('reg', 'Regular')


# In[27]:


train.head(20)


# **Encoding Categorical Variables**
# 

# In[28]:


le = LabelEncoder()


# In[29]:


train['Item_Fat_Content'].unique()


# In[30]:


train['Item_Fat_Content'] = le.fit_transform(train['Item_Fat_Content'])


# In[31]:


train['Item_Visibility_bins'] = le.fit_transform(train['Item_Visibility_bins'])


# In[33]:


train['Outlet_Size'] = le.fit_transform(train['Outlet_Size'])


# In[34]:


train['Outlet_Location_Type'] = le.fit_transform(train['Outlet_Location_Type'])


# In[ ]:


# create dummies for outlet type


# In[35]:


dummy = pd.get_dummies(train['Outlet_Type'])
dummy.head()


# In[36]:


train = pd.concat([train, dummy], axis=1)


# In[37]:


train.dtypes


# In[38]:


# got to drop all the object types features
train = train.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type','Outlet_Establishment_Year'], axis=1)


# In[39]:


train.columns


# In[40]:


train.head()


# **Linear Regression**

# In[41]:


# build the linear regression model
X = train.drop('Item_Outlet_Sales', axis=1)
y = train.Item_Outlet_Sales


# In[42]:


test = pd.read_csv('bigmart_test.csv')
test['Outlet_Size'] = test['Outlet_Size'].fillna('Medium')


# In[43]:


test['Item_Visibility_bins'] = pd.cut(test['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])


# In[44]:


test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].mean())


# In[45]:


test['Item_Visibility_bins'] = test['Item_Visibility_bins'].fillna('Low Viz')
test['Item_Visibility_bins'].head()


# In[46]:


test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace('reg', 'Regular')


# In[47]:


test['Item_Fat_Content'] = le.fit_transform(test['Item_Fat_Content'])


# In[48]:


test['Item_Visibility_bins'] = le.fit_transform(test['Item_Visibility_bins'])


# In[49]:


test['Outlet_Size'] = le.fit_transform(test['Outlet_Size'])


# In[50]:


test['Outlet_Location_Type'] = le.fit_transform(test['Outlet_Location_Type'])


# In[51]:


test['Outlet_Age'] = 2018 - test['Outlet_Establishment_Year']


# In[52]:


dummy = pd.get_dummies(test['Outlet_Type'])
test = pd.concat([test, dummy], axis=1)


# In[53]:


X_test = test.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type','Outlet_Establishment_Year'], axis=1)


# In[54]:


X.columns, X_test.columns


# In[55]:


from sklearn import model_selection
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.3,random_state=42)


# In[56]:


lin = LinearRegression()


# In[57]:


lin.fit(xtrain, ytrain)
print(lin.coef_)
lin.intercept_


# In[ ]:


predictions = lin.predict(xtest)
print(sqrt(mean_squared_error(ytest, predictions)))


# #### A good RMSE for this problem is atleast 1150. 
# 
# Try using Ridge, Lasso, ElasticNet, and compare the RMSE scores.
# You can try Gradient boosting too.
# 
#  **Lesson 08: Ensemble learning** is about xgboost
#  
# Once you have learnt the XGboost techniques, come back and re-work on this problem.
# *XGBOOST* will give the lowest RMSE.
#  
# 

# In[ ]:


from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.001, normalize=True)
ridgeReg.fit(xtrain,ytrain)
print(sqrt(mean_squared_error(ytrain, ridgeReg.predict(xtrain))))
print(sqrt(mean_squared_error(ytest, ridgeReg.predict(xtest))))
print('R2 Value/Coefficient of Determination: {}'.format(ridgeReg.score(xtest, ytest)))


# In[ ]:


from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=0.001, normalize=True)
lassoreg.fit(xtrain, ytrain)

print(sqrt(mean_squared_error(ytrain, lassoreg.predict(xtrain))))
print(sqrt(mean_squared_error(ytest, lassoreg.predict(xtest))))
print('R2 Value/Coefficient of Determination: {}'.format(lassoreg.score(xtest, ytest)))


# In[ ]:


from sklearn.linear_model import ElasticNet
Elas = ElasticNet(alpha=0.001, normalize=True)
Elas.fit(xtrain, ytrain)

print(sqrt(mean_squared_error(ytrain, Elas.predict(xtrain))))
print(sqrt(mean_squared_error(ytest, Elas.predict(xtest))))
print('R2 Value/Coefficient of Determination: {}'.format(Elas.score(xtest, ytest)))

