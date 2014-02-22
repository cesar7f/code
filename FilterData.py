# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:29:51 2014

@author: C
"""

import numpy as np
from NetworkXCreator import NetworkXCreator
import networkx as nx
from scipy.cluster.vq import kmeans,vq
import pandas as pd

class FilterData:
    
    
    
    @staticmethod
    def RANDOM_NODE_FILTER(df_nodes,df_edges,n_nodes=120,time_step=-1):
        
        if(time_step > -1):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            df_edges = df_edges[df_edges.time_step==time_step]
            
        ids = df_nodes.ID.unique()
        ids = np.random.choice(ids,size=n_nodes,replace=False)
        
        df_nodes = df_nodes = df_nodes[df_nodes.ID.isin(ids)]
        df_edges = df_edges[(df_edges.Pred_ID.isin(ids))&(df_edges.Prey_ID.isin(ids))]
        
        G = NetworkXCreator.DF_TO_NETX(df_nodes,df_edges)
        G = nx.connected_component_subgraphs(G)[0]
        
        FilterData.PRINT_BASIC_INFORMATION(G)
        
        return G
        
    @staticmethod   
    def ABUNDANCE_NODE_FILTER(df_nodes,df_edges,min_log_abundance=6,time_step=-1):
        """Filter the data by abundance thresshold"""
        
        if(time_step > -1):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            df_edges = df_edges[df_edges.time_step==time_step]
            
        df_nodes = df_nodes[np.log10(df_nodes.CohortAbundance)>min_log_abundance]
        ids = df_nodes.ID.unique()
        
        df_edges = df_edges[(df_edges.Pred_ID.isin(ids))&(df_edges.Prey_ID.isin(ids))]
        
        Ga = NetworkXCreator.DF_TO_NETX(df_nodes,df_edges)
        G = nx.connected_component_subgraphs(Ga)[0]     
        
        FilterData.PRINT_BASIC_INFORMATION(G)
        

        
        return G
      
    @staticmethod 
    def K_MEANS_NODE_AGG(df_nodes,df_edges,k=80,n_links=2,time_step=-1):
        """Aggregate the data by kmeans thresshold"""
        G = nx.Graph()
        if(time_step > -1):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            df_edges = df_edges[df_edges.time_step==time_step]
        
        data = df_nodes[['JuvenileMass', 'AdultMass']]
        data = data.as_matrix()
        data = np.log10(data)        
        
        centroids,_ = kmeans(data,k)
        idx,_ = vq(data,centroids)
        
        df_nodes['cluster'] = pd.Series(idx, index=df_nodes.index)
        
        edges = []
        maxc = np.max(idx)        
        
        for c1 in range(maxc):
            for c2 in range(maxc+1):
                nodes_a = df_nodes[df_nodes.cluster==c1]['ID'].values
                nodes_b = df_nodes[df_nodes.cluster==c2]['ID'].values
                
                if (len(df_edges[(df_edges.Pred_ID.isin(nodes_a))&(df_edges.Prey_ID.isin(nodes_b))].index) >= n_links):
                    edges.append((c1,c2))
                    continue
                if (len(df_edges[(df_edges.Pred_ID.isin(nodes_b))&(df_edges.Prey_ID.isin(nodes_a))].index) >= n_links):
                    edges.append((c1,c2))
                    continue
                
                
       
        G.add_edges_from(edges)
        
        #Ga = NetworkXCreator.DF_TO_NETX(df_nodes,df_edges)
        #G = nx.connected_component_subgraphs(Ga)[0]     
        G = nx.connected_component_subgraphs(G)[0]
        
        FilterData.PRINT_BASIC_INFORMATION(G)
        

        
        return G
        
    @staticmethod
    def PRINT_BASIC_INFORMATION(G):
        S = 1.0*G.number_of_nodes()
        L = 1.0*G.number_of_edges()
        C = L/(S*S)
        LS = L/S

        print('Nodes = %i' % G.number_of_nodes())
        print('Edges = %i' % G.number_of_edges()) 
        print('Density = %f' % C)
        print('Ls = %f' % LS)
        print('Average Clustering = %f' % nx.average_clustering(G))
        