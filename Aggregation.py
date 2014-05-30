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

    _KMEANS = 1
    _HIERARCHICAL_CLUSTERING = 2    
    
    def __init__(self,nx_graph,algorithm=2):
        
        self._gx              = nx_graph
        self._nclusters       = 30
        self._data            = []
        self._gx_agg_node     = {}        
        self._done           = False
        self._scale_function = np.log10
        self._normalize_data = True
        self._tree           = None
        self._algorithm      = algorithm
        self._distance_matrix= None
        self._tree_done      = False
        self._nodes_ids      = []
        self._clusters_ids   = []
        
    
    def DoClustering(self,nclusters=30):
        '''Main clustering function'''
        
        gx = self._gx; func = self._scale_function
        
        nid,jm,am,fg=zip(*[(x,gx.node[x]['JuvenileMass'],gx.node[x]['AdultMass'],gx.node[x]['FunctionalGroup']) for x in gx.node.keys()])
        data = np.c_[func(jm),func(am)]
        
        if(self._normalize_data==True):
            data = whiten(data)        
        data = np.c_[data,1000*np.array(fg)]
       
        if self._algorithm == Aggregation._HIERARCHICAL_CLUSTERING:

            if not self._tree_done:
                if self._distance_matrix:
                    self._tree = pc.treecluster(distancematrix=self._distance_matrix)
                else:
                    self._tree = pc.treecluster(data)
            
                self._tree_done = True

        self._data = data        
        self._nodes_ids = nid
        clusters_ids = self._tree.cut(nclusters)
        self._clusters_ids = clusters_ids
        self._nclusters = len(np.unique(self._clusters_ids))
        
        cluster_attrib = dict(zip(nid,clusters_ids))
        nx.set_node_attributes(gx,'cluster',cluster_attrib)
        self._gx = gx
        
        for cid in clusters_ids:

            fg = [gx.node[x]['FunctionalGroup'] for x in gx.node.keys() if gx.node[x]['cluster']==cid]
            if len(np.unique(fg)) is not 1:
                raise Exception('Many functional groups inside the same cluster!!!!!! A CRASH JUST HAPPENED, just joking!!!!')
        
        
    def ConstructNodeAggregationMultiGraph(self):
        '''Construct a multigraph with aggregated nodes and all existing edges that existed
        previously. The graph becomes multi graph because now you may have that more than 1
        edge will go in the same direction between two nodes'''
        mgx = nx.MultiDiGraph()
        gx = self._gx
        clusters_id = set(self._clusters_ids)
        n_links = 0
        for cid in clusters_id:
            print cid
            sys.stdout.flush()                 
            nids,jm,am,im,fg,ca = zip(*[(x,gx.node[x]['JuvenileMass'],gx.node[x]['AdultMass'],gx.node[x]['IndividualBodyMass'],
                                    gx.node[x]['FunctionalGroup'],gx.node[x]['CohortAbundance']) for x in gx.node.keys() if gx.node[x]['cluster']==cid])
            
            jm = np.array(jm); am = np.array(am); im = np.array(im);
            fg = np.array(fg); ca = np.array(ca)
            
            attribs = dict()
            attribs['AdultMass'] = np.sum(ca*am)/np.sum(ca)
            attribs['JuvenileMass'] = np.sum(ca*jm)/np.sum(ca)    
            attribs['IndividualBodyMass'] = np.sum(ca*im)/np.sum(ca)
            attribs['ID'] = cid
            attribs['FunctionalGroup'] = int(np.mean(fg))
            attribs['CohortAbundance'] = np.sum(ca)
            attribs['Biomass'] = np.sum(ca*im)             
             
            mgx.add_node(cid,attr_dict = attribs)

            nids = set(nids)
            outgoing_links = [x for x in gx.edges(data=True) if x[0] in nids]
            
            for cid_pred in clusters_id:
                
                try:
                
                    nids_preds = [x for x in gx.node if gx.node[x]['cluster'] == cid_pred]
                    nids_preds = set(nids_preds)
                    to_pred_links = [(cid,cid_pred,x[2]) for x in outgoing_links if x[1] in nids_preds]
                    n_links = len(to_pred_links) + n_links
                    mgx.add_edges_from(to_pred_links)
                except:
                    pdb.set_trace()

        #print 'Number of links in the network is: %i' % n_links

        return mgx            
            
            
            

    def ConstructAggregatedNet(self, agg_flux = True):
        '''Construct aggregated network with flux of individuals as edge weight'''
        mgx = nx.DiGraph()
        gx = self._gx
        clusters_id = set(self._clusters_ids)
        n_links = 0
        nids_clust = dict()
        for cid in clusters_id:
            
            sys.stdout.flush()                 
            nids,jm,am,im,fg,ca = zip(*[(x,gx.node[x]['JuvenileMass'],gx.node[x]['AdultMass'],gx.node[x]['IndividualBodyMass'],
                                    gx.node[x]['FunctionalGroup'],gx.node[x]['CohortAbundance']) for x in gx.node.keys() if gx.node[x]['cluster']==cid])
            
            jm = np.array(jm); am = np.array(am); im = np.array(im);
            fg = np.array(fg); ca = np.array(ca)
            
            attribs = dict()
            attribs['AdultMass'] = np.sum(ca*am)/np.sum(ca)
            attribs['JuvenileMass'] = np.sum(ca*jm)/np.sum(ca)    
            attribs['IndividualBodyMass'] = np.sum(ca*im)/np.sum(ca)
            attribs['ID'] = cid
            attribs['FunctionalGroup'] = int(np.mean(fg))
            attribs['CohortAbundance'] = np.sum(ca)
            attribs['Biomass'] = np.sum(ca*im)             
             
            mgx.add_node(cid,attr_dict = attribs)
        
            nids_clust[cid] = set(nids)
        
        if agg_flux:#individual flux rate is according to aggregated nodes
            edges = nx.get_edge_attributes(gx,'BiomassIngested')
    
            for cid_prey in clusters_id:
                
                nids_prey = nids_clust[cid_prey]
                prey_links = [(x[1],edges[x]) for x in edges if x[0] in nids_prey]
                
                
                for cid_pred in clusters_id:
                    
                    nids_pred = nids_clust[cid_pred]   
                    
                    #shared_links = [edges[x] for x in edges.keys() if x[0] in nids_prey and x[1] in nids_pred]
                    #mass_flow = np.sum(shared_links)
                    shared_links = [y for x,y in prey_links if x in nids_pred]
                    if(len(shared_links) == 0): continue
                
                    mass_flow = np.sum(shared_links)
                    n_links = n_links + len(shared_links)              
                    
                    attribs = dict()        
                    
                    attribs['mass_flow'] = mass_flow
                    attribs['ind_prey_flow'] = mass_flow/mgx.node[cid_prey]['IndividualBodyMass']
                    attribs['weight'] = (mass_flow/mgx.node[cid_prey]['IndividualBodyMass'])/mgx.node[cid_pred]['CohortAbundance']
                    
                    mgx.add_edge(cid_prey,cid_pred,attr_dict=attribs)
        else:
            
            
            edges = nx.get_edge_attributes(gx,'prey_flux')
    
            for cid_prey in clusters_id:
                
                nids_prey = nids_clust[cid_prey]
                prey_links = [(x[1],edges[x]) for x in edges if x[0] in nids_prey]
                
                
                for cid_pred in clusters_id:
                    
                    nids_pred = nids_clust[cid_pred]   
                    
                    #shared_links = [edges[x] for x in edges.keys() if x[0] in nids_prey and x[1] in nids_pred]
                    #mass_flow = np.sum(shared_links)
                    shared_links = [y for x,y in prey_links if x in nids_pred]
                    if(len(shared_links) == 0): continue
                
                    prey_flow = np.sum(shared_links)
                    n_links = n_links + len(shared_links)              
                    
                    attribs = dict()        
                    
                    attribs['prey_flow'] = prey_flow
                    #@attribs['ind_prey_flow'] = mass_flow/mgx.node[cid_prey]['IndividualBodyMass']
                    attribs['weight'] = prey_flow/mgx.node[cid_pred]['CohortAbundance']
                    
                    mgx.add_edge(cid_prey,cid_pred,attr_dict=attribs)
        
        print n_links
        return mgx