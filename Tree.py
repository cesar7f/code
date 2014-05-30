# -*- coding: utf-8 -*-
"""
Created on Mon Mar 03 10:53:05 2014

@author: t-cflore
"""

import numpy as np
#from NetworkXCreator import NetworkXCreator
import networkx as nx
from scipy.cluster.vq import kmeans,vq,whiten
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pdb

class Tree(object):
    
    @staticmethod
    def CONSTRUCT_TREE(df_nodes, df_edges,source=-1,sources=[-1,-2]):
        
        ids = np.union1d(df_edges.Pred_ID.unique(),df_edges.Prey_ID.unique())
        
        df_tree = pd.DataFrame(columns=df_edges.columns)


        for node_id in ids:
            if node_id in sources: continue
            df = df_edges[(df_edges.Pred_ID==node_id) & (df_edges.Prey_ID.isin(sources))].sort('BiomassIngested')
            if len(df) > 0:
                row = df.irow(-1)
            else:
                try:
                    row = df_edges[(df_edges.Pred_ID==node_id)].sort('BiomassIngested').irow(-1)
                except:
                    print 'Discarding node without prey...'
                    print node_id
                    print df_edges[(df_edges.Pred_ID==node_id)].sort('BiomassIngested')
                    #continue                    
                    raise Exception('nodes without prey')
                
            
            df_tree = df_tree.append(row)
                
                
        df_tree[['Pred_ID', 'Prey_ID']] = df_tree[['Pred_ID', 'Prey_ID']].astype(int)
        df_tree[['Biomass_Assimilated','BiomassIngested']] = df_tree[['Biomass_Assimilated','BiomassIngested']].astype(float)
        
        ids = np.union1d(df_tree.Pred_ID.unique(),df_tree.Prey_ID.unique())        
        df_nodes = df_nodes[df_nodes.ID.isin(ids)]                
        
        a_values = {ii:0 for ii in ids}
        c_values = {ii:0 for ii in ids}
        print len(a_values)
        print len(df_nodes)
        print a_values
        print df_tree.Prey_ID.unique()        
        
        df_nodes['A_value'] = a_values.values()
        df_nodes['C_value'] = a_values.values()     
        
        #return df_nodes,df_tree        
        
        Tree.GET_A(df_tree,source,a_values,c_values)
        
        df_nodes = df_nodes.set_index('ID',drop = False)
        
        for key,value in a_values.iteritems():
            #print key,value
            df_nodes.loc[key,'A_value'] = value
        for key,value in c_values.iteritems():
            df_nodes.loc[key,'C_value'] = value

        df_nodes[['A_value','C_value']] = df_nodes[['A_value','C_value']].astype(int)
        return df_nodes,df_tree
           
    @staticmethod
    def GET_A(df_edges,idn,a_values,c_values):
        '''Return some useful metrics when df_nodes and df_edges are in minimal spanning tree format'''
        
        a_values[idn] = 1
        #c_values[idn] = 1
        
        ids = df_edges[df_edges.Prey_ID == idn].Pred_ID
        
        for node_id in ids:
            if a_values[node_id] == 0:
                Tree.GET_A(df_edges,node_id,a_values,c_values)
            else: #This case can not happen in a three
                raise Exception("UPS, your graph is not a tree")
            
            a_values[idn] += a_values[node_id]
            
        c_values[idn] = a_values[idn]
        
        for node_id in ids:
            c_values[idn] += a_values[node_id]
        
        return
        
    

                