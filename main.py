# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:32:02 2018

@author: ASUS
"""

import dataclean
import graphing
import pandas as pd

fname='Dataset - Human Resource.csv'
df=pd.read_csv(fname)
leandf, converteddf, cor, info, rawcorr=dataclean.dataCleanse(df)
graphs=graphing.graph(leandf,converteddf)