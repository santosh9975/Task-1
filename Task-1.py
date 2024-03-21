#!/usr/bin/env python
# coding: utf-8

# # Task 1: Prediction using Supervised ML

# #### Objective : 
#                 Predict the percentage of marks of an student based on the number of study hours
#                 
#                 This is a simple linear regression task as it involves just two variables.
#                 
#                 Data can be found at :http://bit.ly/w-data

# In[4]:


# Importing all libraries required in this notebook 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


# Reading data from remote link
data= pd.read_csv("http://bit.ly/w-data")
data.head(10) 


# In[8]:


# Check shape of data 
data.shape


# In[9]:


data.info()


# In[11]:


# check missing values 
print ("\Missing values : ", data.isnull().sum().values.sum())


# ## There are No missing values are tha dataset

# In[14]:


fig = plt.figure(figsize=(5,5))
plt.subplot(211)
plt.boxplot(data.iloc[:,0])
plt.title("Box Plot of Hours")
plt.subplot(212)
plt.boxplot(data.iloc[:,1])
plt.title("Box Plot of Scores")
plt.show()


# ## There are no outliers in the dataset 

# In[15]:


sns.scatterplot(x = data.Hours, y = data.Scores)


# ##  From the graph above we can clearly see there is a positive linear relation  between the number of hours studied and percentage of scores.

# # Preparing the data

# ####  The next step is to divide the data into "attributes"(inputs) and "lebels" ("outputs)

# In[25]:


x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# #### The next step is to  split this data into training  and test sets We'll do this by using Scikit-Learn's built- in-train_test_split() method

# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# #### Training  the  Alogrithem
# 
# We have split our data into training and testing sets and now id finally the time to train our alogrithem.
# 

# In[27]:


# Train the linear Model on Training Data
Model = LinearRegression()
Model.fit(x_train, y_train)


# In[28]:


# Plotting the regression line 
line = Model.coef_*x+Model.intercept_

# plotting for the test data
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# ### Making Predictions
# Now that we have trained our algorithm , it's time to make some predictions

# In[30]:


y_train_pred = Model.predict(x_train)
print('R2 score of Training Data:',r2_score(y_train, y_train_pred))
y_test_pred = Model.predict(x_test)
print('R2 score of Training Data:',r2_score(y_test, y_test_pred))
print()


# In[32]:


# you can also test with your own data
hours=[[9.25]]
own_pred = Model.predict(hours)
print("No of Hours = {}".format(hours))
print("predicted Score = {}".format(own_pred[0]))


# In[ ]:




