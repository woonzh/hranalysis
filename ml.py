# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:48:24 2018

@author: ASUS
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, metrics
from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score
import dataclean
import numpy as np

trainpercen=0.8
valpercen=0.2
testpercen=0
datasets={}

def splitSets(df):
    global datasets
    df2=shuffle(df)
    
    trainnum=round(trainpercen*len(df2))
    if testpercen==0:
        valnum=len(df2)-trainnum
    else:
        valnum=round(valpercen*len(df2))
    
    datasets['trainx']=df2.loc[list(range(trainnum))].drop(['Attrition'], axis=1)
    datasets['valx']=df2.loc[list(range(trainnum, trainnum+valnum))].drop(['Attrition'], axis=1)
    if testpercen >0:
        datasets['testx']=df2.loc[list(range(trainnum+valnum, len(df2)))].drop(['Attrition'], axis=1)
    
    datasets['trainy']=df2.loc[list(range(trainnum))]['Attrition']
    datasets['valy']=df2.loc[list(range(trainnum, trainnum+valnum))]['Attrition']
    
    if testpercen >0:
        datasets['testx']=df2.loc[list(range(trainnum+valnum, len(df2)))].drop(['Attrition'])
        datasets['testy']=df2.loc[list(range(trainnum+valnum, len(df2)))]['Attrition']
    
    return datasets

def trainLogisticsModel(train, train_labels):
    logreg = LogisticRegression()
    logreg.fit(train, train_labels)
    return logreg

def getAccuracy(model, test, test_label):
    pred = model.predict_proba(test)
    threshold=0.6
    pred=np.array(pred)[:,1]
    pred=(pred>threshold).astype(int)
    
    results=pred+2*test_label
    
    tp=len(results[results==3])
    tn=len(results[results==0])
    fp=len(results[results==1])
    fn=len(results[results==2])
    
    abs_table=pd.DataFrame(columns=['actual negative','actual positive'])
    abs_table.loc['tested negative']=[tn, fn]
    abs_table.loc['tested positive']=[fp, tp]
    
    percen_table=pd.DataFrame(columns=['actual negative','actual positive'])
    percen_table.loc['tested negative']=[tn/(tn+fp), fn/(fn+tp)]
    percen_table.loc['tested positive']=[fp/(tn+fp), tp/(fn+tp)]
    
    print(abs_table)
    print(percen_table)
    print('score: '+str((tp+tn)/len(test)))
    
    return [abs_table,percen_table]

def train(df):
    splitSets(df)
    print('datasets split')
    model=trainLogisticsModel(datasets['trainx'], datasets['trainy'])
    print('model trained')
    results = getAccuracy(model, datasets['valx'],datasets['valy'])
#    print('score: '+str(score))
    
    return results

df=pd.read_csv('Dataset - Human Resource.csv')
leandf, converteddf, cor, info, rawcorr = dataclean.dataCleanse(df)
results=train(converteddf)