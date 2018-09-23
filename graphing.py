# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:33:13 2018

@author: ASUS
"""

import matplotlib.pyplot as mp
import matplotlib.image as mpimg
import pandas as pd
import scipy.cluster.hierarchy as sch
import numpy as np

# open graph based on path
def openGraph(store, col):
    img = mpimg.imread(store[col])
    mp.imshow(img)
    mp.show()

#generate stratas for integer values
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

# identify strata for value
def findStrata(val, strata):
    for i in range(len(strata)):
        if val<=strata.iloc[i,0]:
            return strata.iloc[i,1]

#change numerical column to categorical
def classify(tem):
    store=[]
    strata=generateStrata(tem)
    for i in list(tem):
        store.append(findStrata(i,strata))
        
    return store, strata

# get aggregated values based on categories
def processGraph(data, col):
    tem=data[col]
    if str(tem.dtype)=='object':
        data['class']=list(tem)
    else:
        data['class'], strata=classify(tem)
        
    data=data.groupby(['Attrition', 'class']).count().reset_index()
    return data

#main call
def plotgraph(df):
    store={}
    for col in list(df):
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
                mp.close()
                print(str(col)+' error')
    return store

def plot_corr(df,size, path):
    corr = df.corr()
    
    # Plot the correlation matrix
    fig, ax = mp.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn')
    mp.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    mp.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    path='graphs/corr.png'
    mp.savefig(path)
    mp.close()
    
def corrgraph(df):
    X = df.corr().values
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df2 = df.reindex_axis(columns, axis=1)
    
    path='graphs/corr.png'
    plot_corr(df2, 10, path)
    
    return path

def graph(df, converteddf):
    store=plotgraph(df)
    print('variables graph plotted')
    store['corr']=corrgraph(converteddf)
    print('corr graph plotted')
    
    return store