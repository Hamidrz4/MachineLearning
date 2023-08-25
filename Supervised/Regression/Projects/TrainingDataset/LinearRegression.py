#!/usr/bin/env python
# coding: utf-8

# In[1]:
# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import sklearn.metrics as met 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# In[2]:
# DEFINE ARRAYS (INDEX,COLUMNS)

X = np.array([10,20,30,40,50,60,70,80,90,100]).reshape(-1,1)
y = np.array([18,41,61,79,70,120,141,150,120,200]).reshape(-1,1)

# In[3]:
# DEFINE DATAFRAME

df = pd.DataFrame(X)
df.columns = ["Data"]
df["Target"] = y
df.index=np.arange(1,11)

# In[4]:
# SPLIT DATASET

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.35,random_state=42)

# In[5]:
# PLOT THE POINTS OR -LINES

plt.figure()
plt.plot(xtrain, ytrain,"b.")
plt.plot(xtest, ytest,"r.")
plt.xlabel("Data")
plt.ylabel("Target")
plt.show()

# In[6]:
# DEFINE PREDICTION MODEL

model = LinearRegression()

# In[7]:
# TRAIN MODEL ON TRAIN DATASET

model.fit(xtrain,ytrain)

# In[8]:
# MODEL'S PREDICTION ON TEST DATASET

ypred = model.predict(xtest)

# In[9]:
# PLOT TRAIN VALUES VERSUS PREDICTION VALUES ( marker - legend - linewidth - alpha - fontsize - ls - grid )

plt.Figure()
plt.scatter(xtrain, ytrain,marker="^",color="b",linewidth=5,alpha=0.5)
plt.scatter(xtest, ytest,marker="v",color="r",linewidth=5,alpha=0.5)
plt.legend(["pltTrain","pltTest"])
plt.title("LinearRegression",fontsize=20)
plt.grid(color="green",ls="--",linewidth=1)
plt.xlabel("Data",fontsize=14,color="gray")
plt.ylabel("Target",fontsize=14,color="gray")
plt.plot(xtest,ypred,color="black",ls="dotted", linewidth=5)
plt.show()

# In[10]:
# CREATE REPORT-TRAIN DATAFRAME 

ReportTrain = pd.DataFrame(xtrain)
ReportTrain["TargetTrain"] = ytrain
ReportTrain.columns = ["DataTrain","TargetTrain"]
ReportTrain.index = np.arange(1,7)

# In[11]:
# CREATE REPORT-TEST DATAFRAME ( ASTYPE - RENAME - ARANGE - INDEX&COLUMNS_NAME - SORT_VALUES)

ReportTest = pd.DataFrame(xtest)
ReportTest["PredictionTest"]=ypred
ReportTest["PredictionTest"]=ReportTest["PredictionTest"].astype(int)
ReportTest["TargetTest"]=ytest

ReportTest = ReportTest.rename(columns={ReportTest.columns[0]:'DataTest'})
ReportTest.index = np.arange(1,5)
ReportTest.index.name = 'ID'
ReportTest.columns.name = 'Features'
ReportTest.sort_values(by='DataTest')


# In[12]:
# DATAFRAME DATA TYPE DETAILS 

ReportTest.dtypes

# In[13]:
# GET SPECIFIC COLUMNS WITH THEIR VALUES (DATAFRAME THEME)

ReportTest[['PredictionTest','TargetTest']]

# In[14]:
# ACCESS RECORD BY VALUE ( ID )

ReportTest.loc[[3]]

# In[15]:
# JOIN OR CONCAT TWO DATAFRAMES BY DEFINING KEYS 

#Show = ReportTrain.join(ReportTest,lsuffix='Train',rsuffix='Test',how="outer")
#Show

show = pd.concat([ReportTrain,ReportTest],axis=1,keys=["ReportTrain","ReportTest"])
print(show)

