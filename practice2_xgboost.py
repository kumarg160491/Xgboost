# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:16:12 2019

@author: rajkumar
"""
import pandas as pd
import numpy as np

dataset=pd.read_csv('E:\\data science\\subject vedios\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset.head()
dataset.info()

dataset.dtypes

dataset.columns
dataset.dtypes


#dataset['TotalCharges'].dtypes
#dataset['TotalCharges']=pd.to_numeric(dataset['TotalCharges'])
dataset['TotalCharges'].head()


dataset['TotalCharges'].isna().any()
#dataset['TotalCharges'].


#dataset.dtypes

x=dataset.iloc[:,1:-1]
x.head()
x.columns

#churn=pd.get_dummies(dataset['Churn'],drop_first=True)
#churn=churn.astype('int32')
#churn.head()
#dataset['Churn']=churn
#dataset['Churn'].head()
y=dataset.iloc[:,-1:]
y.head()

x.dtypes

#from sklearn import preprocessing
#prep=preprocessing()
#prep_x=prep.fit(x)

#Gender=pd.get_dummies(dataset['gender'],drop_first=False)
#Gender.head()
#Gender=Gender.astype('int32')


partner=pd.get_dummies(dataset['Partner'],drop_first=True)
partner=partner.astype('int32')
partner.head()
partner.rename()

dependents=pd.get_dummies(dataset['Dependents'],drop_first=True)
dependents=dependents.astype('int32')
dependents.head()

phone_service=pd.get_dummies(dataset['PhoneService'],drop_first=True)
phone_service=phone_service.astype('int32')
phone_service.head()

multipleLines=pd.get_dummies(dataset['MultipleLines'],drop_first=False)
multipleLines=multipleLines.astype('int32')
multipleLines.head()


internetservice=pd.get_dummies(dataset['InternetService'],drop_first=False)
internetservice=internetservice.astype('int32')
internetservice.head()

onlineSecurity=pd.get_dummies(dataset['OnlineSecurity'],drop_first=False)
onlineSecurity=onlineSecurity.astype('int32')
onlineSecurity.head()


onlinebackup=pd.get_dummies(dataset['OnlineBackup'],drop_first=False)
onlinebackup=onlinebackup.astype('int32')
onlinebackup.head()

deviceprotection=pd.get_dummies(dataset['DeviceProtection'],drop_first=False)
deviceprotection=deviceprotection.astype('int32')
deviceprotection.head()

techsupport=pd.get_dummies(dataset['TechSupport'],drop_first=False)
techsupport=techsupport.astype('int32')
techsupport.head()


streamingTV=pd.get_dummies(dataset['StreamingTV'],drop_first=False)
streamingTV=streamingTV.astype('int32')
streamingTV.head()

streamingmovies=pd.get_dummies(dataset['StreamingMovies'],drop_first=False)
streamingmovies=streamingmovies.astype('int32')
streamingmovies.head()


contract=pd.get_dummies(dataset['Contract'],drop_first=False)
contract=contract.astype('int32')
contract.head()

paperlessbilling=pd.get_dummies(dataset['PaperlessBilling'],drop_first=False)
paperlessbilling=paperlessbilling.astype('int32')
paperlessbilling.head()

paymentmethod=pd.get_dummies(dataset['PaymentMethod'],drop_first=False)
paymentmethod=paymentmethod.astype('int32')
paymentmethod.head()
paymentmethod.info()
paymentmethod.columns


#churn=pd.get_dummies(dataset['Churn'],drop_first=True)
#churn=churn.astype('int32')
#churn.head()


x=x.drop(['gender',  'Partner', 'Dependents',        
          'PhoneService', 'MultipleLines', 'InternetService',       
          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',       
          'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',       
          'PaymentMethod'],axis=1)
x.head()

x.dtypes

x=pd.concat([x,Gender,partner,dependents,             
             phone_service,multipleLines,internetservice,             
             onlineSecurity,onlinebackup,             
             deviceprotection,techsupport,streamingTV,             
             streamingmovies, contract,paperlessbilling,paymentmethod],axis=1)
x.head()
#x.dtypes
#x.columns
#x.dtypes
#y=churn
#y.head()

x.dtypes
x.columns
x.info()
y.info()



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


best_feature=SelectKBest(score_func=chi2,k=23)
fit=best_feature.fit(x,y)

dfscore=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)

featurescores=pd.concat([dfcolumns,dfscore],axis=1)
featurescores.columns=['Specs','Score']

featurescores

print(featurescores.nlargest(38,'Score'))


#train and test 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

x_train.head(),x_test.head()

y_train.head(),    y_test.head()

x.info()

#x.dtypes


from xgboost import XGBClassifier

classifier=XGBClassifier()
classifier.fit(x_train,y_train)

y_prid_xgb=classifier.predict(x_test)

#confusion and accuracy metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_prid_xgb)
cm

score=accuracy_score(y_test,y_prid_xgb)
score











