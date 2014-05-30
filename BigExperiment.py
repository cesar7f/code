# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:47:27 2014

@author: t-cflore
"""

import numpy as np
#from NetworkXCreator import NetworkXCreator
import networkx as nx
from scipy.cluster.vq import kmeans,vq,whiten
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pdb as pdb
import Pycluster as pc
from DataManipulation import *
from NetworkSampling import NetworkSampling
from Aggregation import *
import inspect



class BigExperiment(object):
    
    @staticmethod
    def CONNECTANCE_PLOT(gx):
        
        n_values = np.arange(20,1001,40)
        ag = Aggregation(gx)
        pdb.set_trace()
        for n in n_values:
            ag.DoClustering(nclusters = n)        
            mgx = ag.ConstructNodeAggregationMultiGraph()
            
            mgx = NetworkSampling.COMPLETE_SAMPLING(mgx)
            S = mgx.number_of_nodes()
            L = mgx.number_of_edges()
            print S,L,1.0*L/(S**2)
            sys.stdout.flush()   


    @staticmethod
    def P_PLOT(agg_object,p_values = np.arange(0.005,1.1,0.005),ntype=0):
        
        
        slist = []
        llist = []
        clist = []
        for p in p_values:
            
            mgx = NetworkSampling.EDGE_SAMPLING(agg_object,p_edge=p,node_type=ntype,desired_nodes=80)
            #print inspect.getsource(NetworkSampling.EDGE_OUT_SAMPLING)

            #mgx = NetworkSampling.COMPLETE_SAMPLING(mgx)
            S = mgx.number_of_nodes()
            L = mgx.number_of_edges()
            slist.append(S)
            llist.append(L)
            clist.append(1.0*L/(S**2))
            print p,S,L,1.0*L/(S**2)
            sys.stdout.flush()   
            
        return slist,llist,clist