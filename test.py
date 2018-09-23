# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:33:23 2018

@author: woon.zhenhao
"""

import pandas as pd
import matplotlib.pyplot as mp
from sklearn import preprocessing as pp
import math

info={}

def cleanTable(df):
    global info
    transformations={}
    d=list(df.select_dtypes(include=['object']))
    for i in d:
        tem=df[i].astype('category')
        le=pp.LabelEncoder()
        le.fit(tem)
        df[i]=le.transform(tem)
        transformations[i]=le
    
    info['transformations']=transformations
    return df

def findNan(df):
    return df!=df

def filterCorr(df):
    global info
    #find corr
    cor=abs(df.corr()['Attrition'])
    
    #remove Nan columns
    nans=findNan(cor)
    removedColumns=list(nans[nans==True].index)
    df=df.drop(removedColumns,axis=1)
    
    #remove low corr columns
    tem=list(cor[cor<0.01].index)
    removedColumns+=tem
    df=df.drop(tem, axis=1)
    
    info['removed columns']=removedColumns
    return df

def dataCleanse(df):
    df=cleanTable(df)
    df=filterCorr(df)
    
    return df
        
fname='Dataset - Human Resource.csv'
df=pd.read_csv(fname)
df2=dataCleanse(df)