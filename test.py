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
    
def generateStrata(tem):
    classes=10
    strata=pd.DataFrame(columns=['cap','label'])
    minv=min(tem)
    maxv=max(tem)
    interval=(maxv-minv)/(classes-1)
    for i in list(range(1,classes+1)):
        cap=round(minv+(interval*i))
        if i > 1:
            label=str(strata.iloc[i-2,0])+'-'+str(cap)
        else:
            label=str(minv)+'-'+str(cap)
        strata.loc[i]=[cap, label]
    return strata

def findStrata(val, strata):
    for i in range(len(strata)):
        if val<=strata.iloc[i,0]:
            return strata.iloc[i,1]

def classify(tem):
    store=[]
    strata=generateStrata(tem)
    for i in list(tem):
        store.append(findStrata(i,strata))
        
    return store, strata
    
def processGraph(data, col):
    tem=data[col]
    if str(tem.dtype)=='object':
        t=1
    else:
        data['class'], strata=classify(tem)
        data=data.groupby(['Attrition', 'class']).count().reset_index()
        return data

def plotgraph(df):
    store={}
    for col in list(df):
        if str(df[col].dtype)!='object':
            if col != 'Attrition':
                data=processGraph(df[['Attrition',col]], col)
                
                mp.scatter(data['class'], data['Attrition'],s=data[col])
                mp.title('Attrition by '+ str(col))
                mp.xlabel(str(col))
                path='graphs/'+col+'_scatter'+'.png'
                mp.savefig(path)
                mp.close()
                store[str(col)+'_scatter']=path
                
                try:
                    d1=data[data['Attrition']=='Yes']
                    d2=data[data['Attrition']=='No']
                    p1=mp.bar(d1['class'], d1[col], color='r', )
                    p2=mp.bar(d1['class'], d2[col], bottom=d1[col],color='b')
                    mp.title('Attrition by '+ str(col))
                    mp.xlabel(str(col))
                    mp.legend((p1[0], p2[0]), ('Yes', 'No'))
                    path='graphs/'+col+'_bar.png'
                    mp.savefig(path)
                    mp.close()
                    store[str(col)+'_bar']=path
                except:
                    print(str(col)+' error')
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
    global info
    df2=cleanTable(df)
    converteddf, cor=filterCorr(df2)
    leandf=df.drop(info['removed columns'], axis=1)
    
    return leandf, converteddf, cor
        
fname='Dataset - Human Resource.csv'
df=pd.read_csv(fname)
leandf, converteddf, cor=dataCleanse(df)
graphs=plotgraph(leandf)