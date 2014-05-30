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
import pdb as pdb
import Pycluster as pc





class Aggregation(object):
    
    def __init__(self,df_nodes,df_edges):
        
        self._df_edges_clust = []
        self._df_nodes_clust = []
        self._df_nodes       = df_nodes
        self._df_edges       = df_edges
        self._done           = False
        self._n_cluster      = 30
        self._scale_function = np.log10
        self._data           = []
        self._normalize_data = True
    
    def FillClusterIndividualData(self,cluster_index):
        df_nodes = self._df_nodes
        df_edges = self._df_edges
    
        df_nodes['cluster'] = cluster_index
        
        
        
        for i in df_nodes['FunctionalGroup'].unique():
            for j in df_nodes['FunctionalGroup'].unique():
                if j <= i: continue
                
                if len(np.intersect1d(df_nodes[df_nodes['FunctionalGroup']==i]['cluster'].unique(),df_nodes[df_nodes['FunctionalGroup']==j]['cluster'].unique()))>0:
                    raise Exception('Two different functional groups in the same cluster!!!')
                    
        dfni = df_nodes.copy()
        dfni = dfni.set_index('ID',drop=False)
        
        df_connections = df_edges.join(dfni, on='Prey_ID')
        df_connections = df_connections.rename(columns = {'cluster' : 'cluster_prey'})
        df_connections = df_connections[['time_step','Prey_ID','Pred_ID','Biomass_Assimilated','BiomassIngested','cluster_prey']]
        df_connections = df_connections.join(dfni,on='Pred_ID')
        df_connections = df_connections.rename(columns = {'cluster' : 'cluster_pred'})
        df_connections = df_connections[['time_step','Prey_ID','Pred_ID','Biomass_Assimilated','BiomassIngested','cluster_prey','cluster_pred']]
        
        self._done = True
        self._n_clusters = len(np.unique(cluster_index)) + 1
        self._df_edges = df_connections
        self._df_nodes = df_nodes
    
    def FillClusterAggregateData(self):
        
        #NODE CLUSTER    
        
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
        
        #EDGE CLUSTER
        df_connections = self._df_edges.copy()
        df_connections = df_connections[['time_step','Biomass_Assimilated','BiomassIngested','cluster_prey','cluster_pred']]
        df_connections = df_connections.rename(columns = {'cluster_prey':'Prey_ID', 'cluster_pred':'Pred_ID'})
        self._df_edges_clust = df_connections
        
    


            
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
        c_func = self._df_nodes['FunctionalGroup'].values
        centroids = self._centroids
        
        plt.figure(1)
        plt.clf()
        plt.subplot(1,3,1)
        plt.scatter(x, y, c=c_func)
        plt.subplot(1,3,2)
        plt.scatter(x, y, c=c_clust)
        plt.subplot(1,3,3)
        plt.scatter(centroids[:,0], centroids[:,1],c=range(np.max(c_clust)+1))        
    

class KMeansAggregation(Aggregation):
        
            
        
    def __init__(self,df_nodes,df_edges,k=40):
        
        super(KMeansAggregation,self).__init__(df_nodes,df_edges)
        self._k               = k
        self._centroids       = []
        
        
    #def RunEverything(self,k=None):
        
        
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


            
    @staticmethod
    def DO_EVERYTHING(df_nodes,df_edges,k=40):
        km = KMeansAggregation(df_nodes,df_edges,k=k)
        km.DoClustering()
        km.FillDFNodesCluster()
        km.FillDFEdgesCluster()

        return km            
            
    @staticmethod
    def DO_COMPLETE_ANALYSIS(file_nodes='data/small_nodes.csv', file_edges='data/small_edges.csv', k = 40, time_step=None):
        df_nodes,df_edges = DataManipulation.GET_DATASETS(file_nodes, file_edges,time_step)
        kma = KMeansAggregation(df_nodes, df_edges, k)        
        kma.DoClustering()
        kma.PlotClustering()
        
        return kma
        
            
class HierarchicalTreeAggregation(Aggregation):
        
            
        
    def __init__(self,df_nodes,df_edges,nclusters=30,distance_matrix=None):
        
        super(HierarchicalTreeAggregation,self).__init__(df_nodes,df_edges)
        self._tree            = {}
        self._tree_done       = False
        self.DoClustering(nclusters,distance_matrix)
        
        
        
        
    def DoClustering(self,nclusters=30,distance_matrix=None):
        #Avoid working two times
        if not self._tree_done:
            df_nc = self._df_nodes[self._df_nodes['ID']>=0].copy()
            
            data = df_nc[['JuvenileMass', 'AdultMass']]
            data = data.as_matrix()
            data = self._scale_function(data)
    
            if(self._normalize_data==True):
                data = whiten(data)
            
            data = np.c_[data,100.*df_nc.FunctionalGroup.values]
            
            if distance_matrix:
                self._tree = pc.treecluster(distancematrix=distance_matrix)
            else:
                self._tree = pc.treecluster(data)
            
            self._data = data
            self._tree_done = True
        
        self.FillClusterIndividualData(self._tree.cut(nclusters))
                
        

            
    @staticmethod
    def DO_EVERYTHING(df_nodes,df_edges,k=40):
        km = KMeansAggregation(df_nodes,df_edges,k=k)
        km.DoClustering()
        km.FillDFNodesCluster()
        km.FillDFEdgesCluster()

        return km            
            
    @staticmethod
    def DO_COMPLETE_ANALYSIS(file_nodes='data/small_nodes.csv', file_edges='data/small_edges.csv', k = 40, time_step=None):
        df_nodes,df_edges = DataManipulation.GET_DATASETS(file_nodes, file_edges,time_step)
        kma = KMeansAggregation(df_nodes, df_edges, k)        
        kma.DoClustering()
        kma.PlotClustering()
        
        return kma
        
            
        

        
