#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries

import pandas as pd
import sklearn.metrics as met 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[2]:


#import data

data = pd.read_csv(r"C:\Users\Hamidreza\Dropbox\Dataset\diamonds.csv", index_col = 0)


# In[3]:


data.head(5)


# In[4]:

#define swap columns

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

data = swap_columns(data, 'z', 'price')


# In[5]:

# set limited values

data["cut"].value_counts()
cut_map = {"Ideal":5, "Premium":4, "Very Good":3, "Good":2, "Fair":1}
data["cut"] = data["cut"].map(cut_map)
color_map = {"D":7,"E":6,"F":5,"G":4,"H":3,"I":2,"J":1}
data["color"] = data["color"].map(color_map)
clarity_map = {"IF":8,"VVS1":7,"VVS2":6,"VS1":5,"VS2":4,"SI1":3,"SI2":2,"I1":1}
data["clarity"] = data["clarity"].map(clarity_map)


# In[6]:

# Determine X,y

X = data.iloc[:,0:9]
y = data.iloc[:,[9]]

# In[7]:

# set features scale

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[8]:

# split data 

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42)

# In[9]:

# train model

model = LinearRegression()
model.fit(Xtrain,ytrain)
Prediction = model.predict(Xtest)

# In[10]:

# measure error

met.mean_squared_error(ytest,Prediction)

# In[11]:

# plot predicted&Actual-Values

plt.plot(ytest,Prediction,"r.")

