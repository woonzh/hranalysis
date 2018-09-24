# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:32:02 2018

@author: ASUS
"""

import dataclean
import graphing
import pandas as pd
import ml

fname='Dataset - Human Resource.csv'
threshold=0.5 #Results with probability above this threshold will be labelled as attrition = Yes
corrThreshold=0.05 #remove columns below this corr value

df=pd.read_csv(fname)
leandf, converteddf, cor, info, rawcorr=dataclean.dataCleanse(df)
graphs=graphing.graph(leandf,converteddf)
model, results = ml.train(converteddf, threshold)