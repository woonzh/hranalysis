# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:33:23 2018

@author: woon.zhenhao
"""

import pandas as pd
from sklearn import preprocessing as pp

info={}

#convert categorical columns to numerical
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

#find nan values in list
def findNan(df):
    return df!=df

#clean up columns
def filterCorr(df):
    global info
    #find corr
    rawcorr=df.corr()
    store=pd.DataFrame()
    store['cor']=rawcorr['Attrition']
    store['abs cor']=abs(store['cor'])
    store=store.sort_values(by=['abs cor'], ascending=False)
    cor=store['abs cor']
    
    #remove Nan columns
    nans=findNan(cor)
    removedColumns=list(nans[nans==True].index)
    df2=df.drop(removedColumns,axis=1)
    
    #remove low corr columns
    tem=list(cor[cor<0.05].index)
    removedColumns+=tem
    df2=df2.drop(tem, axis=1)
    
    info['removed columns']=removedColumns
    return df2,store, rawcorr

#main call
def dataCleanse(df):
    global info
    df2=cleanTable(df)
    print('table transformed')
    converteddf, cor, rawcorr=filterCorr(df2)
    print('table cleaned based on corr')
    leandf=df.drop(info['removed columns'], axis=1)
    
    return leandf, converteddf, cor, info, rawcorr