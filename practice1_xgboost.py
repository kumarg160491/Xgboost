# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 08:36:33 2019

@author: rajkumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset =pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Hyperparameter-Optimization/master/Churn_Modelling.csv')
dataset.dtypes


## Correlation
import seaborn as sns
#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


x=dataset.iloc[:,3:13]
x.head()
y=dataset.iloc[:,-1:]
y

geography=pd.get_dummies(x['Geography'],drop_first=True)
geography.head()

gender=pd.get_dummies(x['Gender'],drop_first=True)
gender.head()   

x=x.drop(['Geography','Gender'],axis=1)
x.head()


x=pd.concat([x,geography,gender],axis=1)
x.head()


#train and test 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.20,random_state=0)

x_train,x_test


from xgboost import XGBClassifier

xgbc=XGBClassifier()
xgbc.fit(x_train,y_train)

y_prid_xgb=xgbc.predict(x_test)

#confusion and accuracy metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_prid_xgb)
cm

score=accuracy_score(y_test,y_prid_xgb)
score






