# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:19:28 2014

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



class KMeansAggregation(object):
        
            
        
    def __init__(self,df_nodes,df_edges,k=40):
        
        self._df_nodes        = df_nodes
        self._df_nodes_clust  = []
        self._df_edges_clust  = []
        self._df_edges        = df_edges
        self._n_clusters      = 0
        self._done            = 0
        self._clusters        = []
        self._k               = k
        self._data            = []
        self._centroids       = []
        self._normalize_data  = True
        self._scale_function  = np.log10
        
    
        
    def DoClustering(self, k=None, plot_cluster = False):
        
        #Avoid cconsidering the source nodes
        df_nc = self._df_nodes[self._df_nodes['ID']>=0].copy()
        
        if(k):
            self._k = k
            
        k = self._k
        
        data = df_nc[['JuvenileMass', 'AdultMass']]
        data = data.as_matrix()
        data = self._scale_function(data)

        if(self._normalize_data==True):
            data = whiten(data)
        
        data = np.c_[data,100.*df_nc.FunctionalGroup.values]
        
        centroids,_ = kmeans(data,k,iter=100)
        idx,_ = vq(data,centroids)
        
        df_nc['cluster'] = pd.Series(idx, index=df_nc.index)
        
        df_nc.loc[df_nc.index,'cluster']=df_nc['cluster']
        
        for i in df_nc['FunctionalGroup'].unique():
            for j in df_nc['FunctionalGroup'].unique():
                if j <= i: continue
                
                if len(np.intersect1d(df_nc[df_nc['FunctionalGroup']==i]['cluster'].unique(),df_nc[df_nc['FunctionalGroup']==j]['cluster'].unique()))>0:
                    raise Exception('Two different functional groups in the same cluster!!!')
                    
        self._df_nodes['cluster'] = -1
        self._df_nodes.loc[df_nc.index,'cluster'] = df_nc['cluster']
        self._data = data
        self._centroids = centroids
        self._done = True
        self._n_clusters = len(centroids) + 1
        
        if plot_cluster:
            self.PlotClustering()

    def FillDFNodesCluster(self):
        nodes = self._df_nodes[['FunctionalGroup','cluster','AdultMass','JuvenileMass','IndividualBodyMass','CohortAbundance']]
        nodes_cluster = nodes.groupby('cluster')
        
        def nodes_aggregation(nodes):
            adult_mass_values = nodes['AdultMass'].values
            juvenile_mass_values = nodes['JuvenileMass'].values
            individual_mass_values = nodes['IndividualBodyMass'].values
            cohort_abundance = nodes['CohortAbundance'].values
            
            dic = dict()
            dic['AdultMass'] = np.sum(cohort_abundance*adult_mass_values)/np.sum(cohort_abundance)
            dic['JuvenileMass'] = np.sum(cohort_abundance*juvenile_mass_values)/np.sum(cohort_abundance)    
            dic['IndividualBodyMass'] = np.sum(cohort_abundance*individual_mass_values)/np.sum(cohort_abundance)
            dic['ID'] = int(np.mean(nodes['cluster']))
            dic['FunctionalGroup'] = int(np.mean(nodes['FunctionalGroup']))
            dic['CohortAbundance'] = np.sum(cohort_abundance)
            dic['Biomass'] = np.sum(cohort_abundance*individual_mass_values)
            
            #dic['log_adult_mass'] = np.log10(dic['AdultMass'])
            #dic['log_juv_mass'] = np.log10(dic['JuvenileMass'])
            
            #dic['log_biomass'] = np.log10(dic['Biomass'])
            #dic['log_individual_mass'] = np.log10(dic['IndividualBodyMass'])
               
            return pd.Series(dic)
              
        df_nodes_cluster = nodes_cluster.apply(nodes_aggregation)
        df_nodes_cluster[['FunctionalGroup','ID']] = df_nodes_cluster[['FunctionalGroup','ID']].astype(int)
        
        self._df_nodes_clust = df_nodes_cluster
        
    def FillDFEdgesCluster(self):
        edges = []
        df_nodes = self._df_nodes
        df_edges = self._df_edges
        clusters = np.sort(self._df_nodes['cluster'].unique())
        df_nodes_cluster = self._df_nodes_clust
        
        nodes_work = dict()
        for i in clusters:
            nodes_work[i] = df_nodes[df_nodes['cluster']==i][['ID','IndividualBodyMass','CohortAbundance']]
            nodes_work[i].sort('ID')
            nodes_work[i] = nodes_work[i].set_index('ID',drop=False)
            
        
        
        for c_pred in clusters:
            print 'Predator %i' % c_pred
            pred_nodes = nodes_work[c_pred]
            sys.stdout.flush()
            #pred_nodes.sort('ID')
            
            for c_prey in clusters:
                
                prey_nodes = nodes_work[c_prey]
                #prey_nodes.sort('ID')
                
                df_connections = df_edges[(df_edges['Pred_ID'].isin(pred_nodes['ID']))&(df_edges['Prey_ID'].isin(prey_nodes['ID']))]
                
                if (len(df_connections.index) >= 1):
                    
                    if(c_pred==-1):
                        pdb.set_trace()
                        return pred_nodes,df_connections
                    ingest_mass_flow = df_connections['BiomassIngested'].sum()
                    ingest_mass_prey_ratio = ingest_mass_flow/(df_nodes_cluster['Biomass'][c_prey])
                    ingest_mass_pred_ratio = ingest_mass_flow/(df_nodes_cluster['Biomass'][c_pred])
                    
                    assim_mass_flow = df_connections['Biomass_Assimilated'].sum()
                    assim_mass_prey_ratio = assim_mass_flow/(df_nodes_cluster['Biomass'][c_prey])
                    assim_mass_pred_ratio = assim_mass_flow/(df_nodes_cluster['Biomass'][c_pred])
                    
                    edges.append((c_pred,c_prey, 
                                  ingest_mass_flow, ingest_mass_prey_ratio, ingest_mass_pred_ratio,
                                  assim_mass_flow, assim_mass_prey_ratio, assim_mass_pred_ratio))
                    
        df_edges_cluster = pd.DataFrame(edges,columns=['Pred_ID','Prey_ID',
                                                       'BiomassIngested','prey_ratio_BiomassIngested','pred_ratio_BiomassIngested',
                                                       'Biomass_Assimilated','prey_ratio_Biomass_Assimilated','pred_ratio_Biomass_Assimilated'])
        #df_edges_cluster.to_csv('big_edges_cluster_30jan.csv')
        self._df_edges_clust = df_edges_cluster
        

    
        
        
        
    
    def PlotClustering(self):
        
        x = self._data[:,0]
        y = self._data[:,1]
        c_clust = self._df_nodes['cluster']
        c_func = self._df_nodes['FunctionalGroup']
        centroids = self._centroids
        
        plt.figure(1)
        plt.clf()
        plt.subplot(1,3,1)
        plt.scatter(x, y, c=c_func)
        plt.subplot(1,3,2)
        plt.scatter(x, y, c=c_clust)
        plt.subplot(1,3,3)
        plt.scatter(centroids[:,0], centroids[:,1],c=range(np.max(c_clust)+1))        
        
            
    @staticmethod
    def DO_COMPLETE_ANALYSIS(file_nodes='data/small_nodes.csv', file_edges='data/small_edges.csv', k = 40, time_step=None):
        df_nodes,df_edges = DataManipulation.GET_DATASETS(file_nodes, file_edges,time_step)
        kma = KMeansAggregation(df_nodes, df_edges, k)        
        kma.DoClustering()
        kma.PlotClustering()
        
        return kma
        
            
        
