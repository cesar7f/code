# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:02:15 2014

@author: t-cflore
"""



execfile('code/KMeansAggregation.py')
execfile('code/BasicExperiments.py')
execfile('code/FilterData.py')
execfile('code/NetworkXCreator.py')


df_nodes = pd.read_csv('data/small_nodes.csv')
df_edges = pd.read_csv('data/small_edges.csv')
