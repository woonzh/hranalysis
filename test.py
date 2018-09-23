# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:33:23 2018

@author: woon.zhenhao
"""

import pandas as pd
import matplotlib.pyplot as mp
import matplotlib.image as mpimg
from sklearn import preprocessing as pp

info={}

def openGraph(store, col):
    img = mpimg.imread(store[col])
    mp.imshow(img)
    mp.show()

def plotgraph(df):
    store={}
    for col in list(df):
        if col != 'Attrition':
            mp.scatter(df[col], df['Attrition'])
            path='graphs/'+col+'.png'
            mp.savefig(path)
            store[col]=path
    return store

def cleanTable(df):
    global info
    transformations={}
    df2=df.copy()
    d=list(df2.select_dtypes(include=['object']))
    for i in d:
        tem=df2[i].astype('category')
        le=pp.LabelEncoder()
        le.fit(tem)
        df2[i]=le.transform(tem)
        transformations[i]=le
    
    info['transformations']=transformations
    return df2

def findNan(df):
    return df!=df

def filterCorr(df):
    global info
    #find corr
    store=pd.DataFrame()
    store['cor']=df.corr()['Attrition']
    store['abs cor']=abs(store['cor'])
    cor=df.corr()['Attrition']
    store=store.sort_values(by=['abs cor'], ascending=False)
    cor=store['abs cor']
    
    #remove Nan columns
    nans=findNan(cor)
    removedColumns=list(nans[nans==True].index)
    df2=df.drop(removedColumns,axis=1)
    
    #remove low corr columns
    tem=list(cor[cor<0.1].index)
    removedColumns+=tem
    df2=df.drop(tem, axis=1)
    
    info['removed columns']=removedColumns
    return df2,store

def dataCleanse(df):
    df2=cleanTable(df)
    df2, cor=filterCorr(df2)
    
    return df2, cor
        
fname='Dataset - Human Resource.csv'
df=pd.read_csv(fname)
df2, cor=dataCleanse(df)
graphs=plotgraph(df)