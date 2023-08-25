#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libreries(load_iris, f1_score, confusion_matrix, mlxtend.plotting)
import os,sys ,seaborn as sns 
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


# In[2]:


#load data from array (nunique, ravel)

df = load_iris()
x = df["data"]
y = df["target"]
#x = pd.DataFrame(x)
#y = pd.DataFrame(y)

#y.nunique()
#y.value_counts()
#y.describe().T
#y.info()
#show = pd.concat([x,y],axis=1,keys=['x','y'])
#y = np.array(y)
#y = y.ravel()
x


# In[3]:


#train model

model = LogisticRegression(solver="lbfgs",max_iter =500, multi_class="multinomial",penalty="l2")

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.30,random_state=21)

model.fit(xtrain,ytrain)

pdn = model.predict(xtest)


# In[4]:


#plot confusion_matrix
sns.set_palette(sns.colorpalette())
, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(confusion_matrix(ytest, pdn), annot=True, fmt='d', annot_kws={"size": 40, "weight": "bold"})
labels = ['False', 'True']
ax.set_xticklabels(labels, fontsize=25);
ax.set_yticklabels(labels[::-1], fontsize=25);
ax.set_ylabel('Prediction', fontsize=30);
ax.set_xlabel('Ground Truth', fontsize=30)

cm = confusion_matrix(ytest,pdn)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4,4))
plt.xlabel("prediction")
plt.ylabel("actual")
plt.title("Confusion Matrix")
plt.show()


# In[5]:


#report statistics

FinalReport = classification_report(ytest,pdn)

Accuracy = accuracy_score(ytest,pdn)

F1_score = f1_score(ytest,pdn,average="micro")

print(FinalReport)

print(f"Final Accuracy is {round(Accuracy,2)}")

print(f"Final F1_score is {round(F1_score,2)}")





