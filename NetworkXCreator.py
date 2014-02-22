# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\t-cflore\.spyder2\.temp.py
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class NetworkXCreator:
    
    @staticmethod
    def DF_TO_NETX(df_nodes,df_edges,directed=False):
        """Return a network x object using nodes and edges data frame"""
    
        if (directed == False):
            G = nx.Graph()
        else:
            G = nx.DiGraph()
        
        G.add_edges_from(zip(df_edges.Pred_ID,df_edges.Prey_ID))    
        return G
    
    @staticmethod
    def CSV_TO_NETX(prefix_name, time_step=-1,directed=False):
        #Creates a Networkx from CSV files with the appropiate format
        
        node_file_name = prefix_name+'_nodes.csv'
        edge_file_name = prefix_name+'_edges.csv'
        
        df_nodes = pd.read_csv(node_file_name)
        df_edges = pd.read_csv(edge_file_name)
        
        #if time step is defined filter the data according to time_step
        if(time_step>-1):
            df_nodes = df_nodes[df_nodes.TimeStep == time_step]
            df_edges = df_edges[df_edges.time_step == time_step]
        
        return NetworkXCreator.DF_TO_NETX(df_nodes,df_edges,directed=False)
    
    @staticmethod
    def GET_NETX_TIME_LAYERS(prefix_name, directed=False, time_layers=-1,name=''):
        #Get all specific time layers as separated graphs
        
        node_file_name = prefix_name+'_nodes.csv'
        edge_file_name = prefix_name+'_edges.csv'
        
        df_nodes = pd.read_csv(node_file_name)
        df_edges = pd.read_csv(edge_file_name)
        
        df_nodes = df_nodes[df_nodes.ID >=0] # Ignore infinity data
        df_nodes[['CohortAbundance']] = df_nodes[['CohortAbundance']].astype(float)        
        
        if time_layers == -1:
            time_ids = df_edges.time_step.unique()
            time_ids.sort()
        else:
            time_ids = time_layers
        
        Gs = []
        
        for t_id in time_ids:
            df_n = df_nodes[df_nodes.TimeStep == t_id]
            df_e = df_edges[df_edges.time_step == t_id]
            
            print 'Reading time step = %i' % t_id
            G = NetworkXCreator.DF_TO_NETX(df_n,df_e,directed=directed)
            G.name = '%s_t_%i' % (name,t_id)
            Gs.append(G)
        
        return Gs
        