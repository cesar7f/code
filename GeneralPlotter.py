# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:38:57 2014

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

class GeneralPlotter:

    _FG_RGB_COLORS = {10:[0,100,0],
                      11:[255,0,0],
                      12:[0,0,100],
                      13:[180,255,180],
                      14:[255,180,180],
                      15:[180,180,255],
                      16:[0,255,0],
                      17:[100,0,0],
                      18:[0,0,255],
                      100:[0,0,0]} # Source Functional group

    @staticmethod
    def PLOT_ADULT_VS_JUVENILE_MASS(df_nodes,fontaxis=14):
        my_colors = []
        df_nodes = df_nodes[df_nodes.FunctionalGroup <= 20]
        for fg in df_nodes.FunctionalGroup:
            my_colors.append(np.array(GeneralPlotter._FG_RGB_COLORS[fg])/255.0)
            
        ssmin = 10
        ssmax = 80
        ss = df_nodes.CohortAbundance.values*df_nodes.IndividualBodyMass.values
        ss = ssmin + (ss-np.min(ss))*(ssmax-ssmin)/(np.max(ss)-np.min(ss))
         
        plt.scatter(np.log10(df_nodes.AdultMass),np.log10(df_nodes.JuvenileMass),c=my_colors,s=ss, alpha=0.6)
        plt.xlabel('log10 (Adult Mass)',fontsize=fontaxis)
        plt.ylabel('log10 (Juvenile Mass)',fontsize=fontaxis)
        
        
    @staticmethod
    def PLOT_PRED_VS_PREY(df_nodes,df_edges,field_name = 'IndividualBodyMass',log_filter=1000):
        df_nodes = df_nodes[df_nodes.FunctionalGroup <= 30]
        df_nodes['Biomass'] = df_nodes['IndividualBodyMass']*df_nodes['CohortAbundance']
        df_edges = df_edges[df_edges.Pred_ID >= 0]
        df_edges = df_edges[df_edges.Prey_ID >= 0]
        df_edges[df_edges['Prey_ID'].isin(df_nodes.ID.unique())]
        df_edges[df_edges['Pred_ID'].isin(df_nodes.ID.unique())]
        df_edges = df_edges[np.log10(df_edges.BiomassIngested) > log_filter]
        df_edges = df_edges[['Prey_ID','Pred_ID']]
        
        dfni = df_nodes.copy()
        dfni = dfni.set_index('ID',drop=False)
        
        df_mass = df_edges.join(dfni, on='Prey_ID')
        df_mass = df_mass.rename(columns = {field_name : 'mass_prey'})
        df_mass = df_mass[['Prey_ID','Pred_ID','mass_prey']]
        df_mass = df_mass.join(dfni, on='Pred_ID')
        df_mass = df_mass.rename(columns = {field_name : 'mass_pred'})
        df_mass = df_mass[['Pred_ID','Prey_ID','mass_pred','mass_prey']]
        
        xx = [df_mass['mass_pred'].min(),df_mass['mass_pred'].max()]
        yy = [df_mass['mass_prey'].min(),df_mass['mass_prey'].max()]
              
        plt.loglog(df_mass['mass_pred'],df_mass['mass_prey'],'ro')
        plt.loglog(xx,yy)
        plt.xlabel('Predator %s' % field_name,fontsize=16)
        plt.ylabel('Prey %s' % field_name,fontsize=16)
                  
           
           
    @staticmethod
    def MASS_RATIO_DISTANCE(u,v):
        '''Similarity measure based in mass ratio. Similar to what Mandigley is
        doing for the mergint process''' 
        if u[2]==v[2]:
            diff_vec = 1.0*np.abs(u[:2]-v[:2])/np.minimum(u[:2],v[:2])
            return np.sqrt((diff_vec**2).sum())
        else:
            return np.inf
        
        
                
    @staticmethod
    def IMPROVED_K_MEANS(df_nodes,df_edges, k=80):
        df_nc = df_nodes.copy()
        
        
        k=90
        data = df_nc[['JuvenileMass', 'AdultMass']]
        data = data.as_matrix()
        
        data = np.log10(data)
        data = whiten(data)
        data = np.c_[data,1000.*df_nodes.FunctionalGroup.values]
        
        centroids,_ = kmeans(data,k)
        idx,_ = vq(data,centroids)
        
        df_nc['cluster'] = pd.Series(idx, index=df_nc.index)
        
        x = data[:,0]
        y = data[:,1]
        c_clust = df_nc['cluster']
        c_func = df_nc['FunctionalGroup']
        
        
        plt.figure(1)
        plt.clf()
        plt.subplot(1,3,1)
        plt.scatter(x, y, c=c_func)
        plt.subplot(1,3,2)
        plt.scatter(x, y, c=c_clust)
        plt.subplot(1,3,3)
        plt.scatter(centroids[:,0], centroids[:,1],c=range(np.max(c_clust)+1))
        
        df_nodes['cluster']=-1
        df_nodes.loc[df_nc.index,'cluster']=df_nc['cluster']
        
        for i in df_nodes['FunctionalGroup'].unique():
            for j in df_nodes['FunctionalGroup'].unique():
                if j <= i: continue
                if len(numpy.intersect1d(aa.loc[i]['cluster'].unique(),aa.loc[j]['cluster'].unique()))>0:
                    print 'Something bad happened'
        
        n_links = 3
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
                    edges.append((c2,c1))
                    continue        
                    
        #G = nx.DiGraph()
        #G.add_edges_from(edges)
                
        #Ga = NetworkXCreator.DF_TO_NETX(df_nodes,df_edges)
        #G = nx.connected_component_subgraphs(Ga)[0]     
        #G = nx.connected_component_subgraphs(G)[0]
        
        #FilterData.PRINT_BASIC_INFORMATION(G)
        #nx.write_gexf(G,'my_hope.gexf')

                
                
                    
        
        
    @staticmethod
    def PLOT_PREY_VS_PRED_MASS(df_nodes, df_edges, time_step=-1, f_group=-1):
        
        if(time_step > -1):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            df_edges = df_edges[df_edges.time_step==time_step]
            
        if f_group > -1: 
            df_nodes = df_nodes[df_nodes['FunctionalGroup']==f_group]
        
        df_edges = df_edges[df_edges.Prey_ID.isin(df_nodes.ID.unique())]
        
        xx = [df_nodes.IndividualBodyMass[df_nodes.ID==x].values[0] for x in df_edges.Pred_ID]
        yy = [df_nodes.IndividualBodyMass[df_nodes.ID==x].values[0] for x in df_edges.Prey_ID]
        #xx = df_nodes.IndividualBodyMass[df_edges.Pred_ID]
        #yy = df_nodes.IndividualBodyMass[df_edges.Prey_ID]
        #df_nodes.reindex(columns=['ID'])
        
        #xx = df_nodes.ix[df_edges['Pred_ID']]['IndividualBodyMass']
        #yy = df_nodes.ix[df_edges['Prey_ID']]['IndividualBodyMass']


        plt.figure(1)
        plt.clf()
        plt.loglog(xx, yy,'bo')


    @staticmethod
    def PLOT_DENSITY_MASS(df_nodes,time_step=-1,f_group=-1):

        if(time_step > -1):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            #df_edges = df_edges[df_edges.time_step==time_step]
            
        if f_group > -1: df_nodes = df_nodes[df_nodes['FunctionalGroup']==f_group]
        
        x = df_nodes['AdultMass']   
        y = df_nodes['JuvenileMass']
        #y = df_nodes['CohortAbundance']
        
        class_id = df_nodes['FunctionalGroup']
        
        plt.figure(1)
        plt.clf()
        plt.subplot(1,2,1)
        plt.scatter(x, y, c=class_id)
        plt.subplot(1,2,2)
        plt.scatter(np.log10(x), np.log10(y), c=class_id)
        #plt.legend(tuple([str(label) for label in class_id.unique()]))

        
    
        
    
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
        