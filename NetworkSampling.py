# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 12:18:47 2014

@author: t-cflore
"""

import pandas as pd
import lxml.etree as etree
import numpy as np
from os import listdir
from os.path import isfile, join
#from IPython.parallel import Client
from multiprocessing import Pool
import sys
import igraph as ig
import networkx as nx
import pdb



class NetworkSampling(object):
    
    @staticmethod
    def GIANT_COMPONENT(gg):
        xmax = 0
        vec = []
        for x in gg.components(mode=1):
            if len(x) > xmax:
                xmax = len(x)
                vec = x
        return gg.subgraph(vec)   
        
    @staticmethod
    def WEIGHETD_RANDOM_CHOICE(choices,n_choices):
            max = sum(choices.values())
            pick = np.random.uniform(0, max,n_choices)
            current = 0
            selection = [None] * n_choices
            picked_value = 0
            for ii in range(n_choices):
                current = 0
                for key, value in choices.items():
                    current += value
                    if current > pick[ii]:
                        picked_value = key
                        break
                selection[ii] = picked_value
                
            return selection
    
    @staticmethod
    def FITNESS_PROBABILITY(values):        
        min_values = np.min(values)        
        if min_values < 0:
            values = values + 2*np.abs(min_values)
        return values/np.sum(values)


    @staticmethod    
    def GET_RANDOM_CHOICE(abundance_vector,n,replace=True):
        '''Select n random individuals given an abundance vector'''
    
        x = [np.ceil(y) for y in abundance_vector]
        
        return np.random.choice(range(len(x)),size=n,replace=replace,p=NetworkSampling.FITNESS_PROBABILITY(x) )
    
    
    
    @staticmethod
    def RANDOM_NODE_SELECTION(graph_object, weights=None, source_node = -1, desired_nodes = 100, desired_links= -1,map_function = np.log10):
        '''
        A GraphObject instance in which run sampling by random node selection. The cohort abundance is the
        field that is used for bias election. However the user can use other objects.
        '''
        #nodes = {v.index:np.log10(v['CohortAbundance']) for v in ig_graph.vs if v.index is not 0}
        nodes = graph_object.GetNodesAttribute('CohortAbundance',map_function=np.log10)
        nodes_selection = np.random.choice(nodes.keys(),size=desired_nodes,replace=False,p=NetworkSampling.FITNESS_PROBABILITY(nodes.values()) )
        
        graph_object = graph_object.SubGraphByNodes(nodes_selection)
        graph_object = graph_object.GetGiantComponent()
        
        if(desired_links < 0):
            desired_links = int(0.41*desired_nodes**1.57)
            
        if desired_links > graph_object.NumberOfEdges():
            desired_links = graph_object.NumberOfEdges()
            
        #edges = {e.index:np.log10(e['BiomassIngested']) for e in ig_graph.es}
        edges = graph_object.GetEdgesAttribute('BiomassIngested',map_function=np.log10)
        edges_selection = np.random.choice(range(len(edges.keys())),size=desired_links,replace=False,p=NetworkSampling.FITNESS_PROBABILITY(edges.values()) )
        edges_selection = [edges.keys()[ii] for ii in edges_selection]        
        
        graph_object = graph_object.SubGraphByEdges(edges_selection)
        graph_object = graph_object.GetGiantComponent()
        
        return graph_object
     
    @staticmethod
    def COMPLETE_SAMPLING(gx):
        '''Return complete sampling'''
        gx_sampled = nx.DiGraph()
        nodes = gx.nodes()
        
        for node_prey in nodes:
            for node_pred in nodes:
                if gx.get_edge_data(node_prey,node_pred) is not None:
                    gx_sampled.add_edge(node_prey,node_pred)
        
        return gx_sampled            
        
    @staticmethod
    def EDGE_SAMPLING(agg_object,p_edge=0.10,node_type=0,desired_nodes=1):
        '''Return edge sampling with probability p_edge. All nodes are conserved'''
        
        gx_sampled = nx.DiGraph()
        gx = agg_object._gx
        clusters_id = set(agg_object._clusters_ids)
        n_links = 0
        #pdb.set_trace()
        for cid in clusters_id:
            #print cid
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
             
            gx_sampled.add_node(cid,attr_dict = attribs)

        nodes = nx.get_node_attributes(gx_sampled,'CohortAbundance')
        nodes_selection = np.random.choice(nodes.keys(),size=desired_nodes,replace=False,p=NetworkSampling.FITNESS_PROBABILITY(np.log10(nodes.values())))
        nodes_selected = [ids for ids in gx.node.keys() if gx.node[ids]['cluster'] in nodes_selection]

        gx2=gx.subgraph(nodes_selected)
        gx_sampled = gx_sampled.subgraph(nodes_selection)

        edges = nx.get_edge_attributes(gx2,'BiomassIngested')
        #pdb.set_trace()
        for cid in nodes_selection:
            
            nids = [x for x in gx2.node if gx2.node[x]['cluster'] == cid]
            nids = set(nids)
            #outgoing_links = [x for x in edges if x[0] in nids_prey]
            if node_type == 0:
                try:
                    s,t,b=zip(*[(x[0],x[1],edges[x]) for x in edges.keys() if x[0] in nids])
                except:
                    print [(x[0],x[1],edges[x]) for x in edges.keys() if x[0] in nids]
                    continue
                nl = len(s)            
                
                selected = 0
                if nl > 0:
                
                    ne = int(nl*p_edge)
                    weights = NetworkSampling.FITNESS_PROBABILITY(np.log10(b))
                    selected = np.random.choice(range(nl), size=ne, replace=False, p = weights)
                    targets = np.unique([gx2.node[t[x]]['cluster'] for x in selected])
                    for tar in targets:
                        gx_sampled.add_edge(cid,tar)
                    
            else:#links pointing to node
                try:
                    s,t,b=zip(*[(x[0],x[1],edges[x]) for x in edges.keys() if x[1] in nids])
                except:
                    [(x[0],x[1],edges[x]) for x in edges.keys() if x[1] in nids]
                    continue
                nl = len(s)            
                
                selected = 0
                if nl > 0:
                
                    ne = int(nl*p_edge)
                    weights = NetworkSampling.FITNESS_PROBABILITY(np.log10(b))
                    selected = np.random.choice(range(nl), size=ne, replace=False, p = weights)
                    targets = np.unique([gx2.node[s[x]]['cluster'] for x in selected])
                    for tar in targets:
                        gx_sampled.add_edge(tar,cid)
                    
            n_links += nl
            
       #print 'Number of links in the network is: %i' % n_links

        return gx_sampled           

    @staticmethod
    def RANDOM_EDGE_SELECTION(nx_graph,df_nodes=None, df_edges=None, desired_nodes = 100, desired_links= 1000):
        
        edges = {e.index:np.log10(e['BiomassIngested']) for e in ig_graph.es}
        edges_selection = NetworkSampling.WEIGHETD_RANDOM_CHOICE(edges,desired_links)
        edges_selection = np.unique(edges_selection)        
        
        ig_graph = ig_graph.subgraph_edges(edges_selection)   
        
    @staticmethod
    def REAL_NETWORK_SAMPLING(mgx,fraction=1.0/30,nsampling=1000):
        '''Return a sampling network assuming empirical sampling'''
        nodes = mgx.nodes()        
        matrix_transfer = nx.adjacency_matrix(mgx,nodelist=nodes)
        
        abundance = nx.get_node_attributes(mgx,'CohortAbundance')
        
        selection=np.random.choice(range(n),p=NetworkSampling.FITNESS_PROBABILITY(abundance),size=nsampling)
        
        nodes = nx.get_node_attributes(gx_sampled,'CohortAbundance')
        
    @staticmethod
    def GET_SAMPLED_MATRIX(pmatrix):
        '''Get a boolean sampled matrix given a probability matrix'''
        nr,nc = pmatrix.shape
        assert(nr==nc)
        
        random_matrix = np.random.rand(nr,nr)
        return random_matrix < pmatrix
    
    @staticmethod
    def GET_SAMPLED_NX(pmatrix=[[1,1],[0,0]],matrix_transfer = None, sampling_size = 100, tau = 1.0/30, giant_component=True):
        '''Get a sampled nx given a probability interaction matrix'''
        if matrix_transfer is not None:
            pmatrix = NetworkSampling.GET_PMATRIX(matrix_transfer,sampling_size=sampling_size,tau=tau)
            
        pmatrix = np.asarray(pmatrix)
        sampled_matrix = NetworkSampling.GET_SAMPLED_MATRIX(pmatrix)
        gx = nx.from_numpy_matrix(np.asmatrix(sampled_matrix),create_using=nx.DiGraph())
        
        if giant_component:
            gx = nx.weakly_connected_component_subgraphs(gx)[0]
            
        return gx
            
    @staticmethod
    def GET_SAMPLED_IG(pmatrix=[[1,1],[0,0]],matrix_transfer = None, sampling_size = 100, tau = 1.0/30, giant_component=True):
        '''Get a sampled nx given a probability interaction matrix'''
        print sampling_size
        if matrix_transfer is not None:
            pmatrix = NetworkSampling.GET_PMATRIX(matrix_transfer,sampling_size=sampling_size,tau=tau)
            
        pmatrix = np.asarray(pmatrix)
        sampled_matrix = list(1.0*NetworkSampling.GET_SAMPLED_MATRIX(pmatrix))
        gi = ig.Graph.Adjacency(sampled_matrix)
        
        if giant_component:
            xmax = 0
            vec = []
            for x in gi.components(mode=1):
                if len(x) > xmax:
                    xmax = len(x)
                    vec = x
            gi = gi.subgraph(vec)    
        return gi

        
    @staticmethod
    def GET_PMATRIX(matrix_transfer, sampling_size=100, tau = 1.0/30):
        '''Get a probability interaction matrix given a sampling predator size
        A poisson process is used with trasnfer rates indicated in matrix_transfer,
        and tau as time to test'''
        pmatrix = 1 - np.exp(-tau*sampling_size*matrix_transfer)
        print sampling_size
        return pmatrix

#        nodes = {v.index:np.log10(v['CohortAbundance']) for v in ig_graph.vs if v.index is not 0}
#        
#        nodes_selection = NetworkSampling.WEIGHETD_RANDOM_CHOICE(nodes,desired_nodes)
#        nodes_selection = np.unique(nodes_selection)
#        nodes_selection = np.append(nodes_selection, 0)
#        
#        ig_graph = NetworkSampling.GIANT_COMPONENT(ig_graph.subgraph(nodes_selection))
#        def random_walk(graph, start_node=None, size=-1, metropolized=False):    
#    """
#    random_walk(G, start_node=None, size=-1):
#    
#    Generates nodes sampled by a random walk (classic or metropolized)
#    
#    Parameters
#    ----------  
#    graph:        - networkx.Graph 
#    start_node    - starting node (if None, then chosen uniformly at random)
#    size          - desired sample length (int). If -1 (default), then the generator never stops
#    metropolized  - False (default): classic Random Walk
#                    True:  Metropolis Hastings Random Walk (with the uniform target node distribution) 
#    """
#    
#    #if type(graph) != nx.Graph:
#    #    raise nx.NetworkXException("Graph must be a simple undirected graph!") 
#        
#    if start_node==None:
#        start_node = random.choice(graph.nodes())
#    
#    v = start_node
#    for c in itertools.count():
#        if c==size:  return
#        if metropolized:   # Metropolis Hastings Random Walk (with the uniform target node distribution) 
#            candidate = random.choice(graph.neighbors(v))
#            v = candidate if (random.random() < float(graph.degree(v))/graph.degree(candidate)) else v
#        else:              # classic Random Walk
#            v = random.choice(graph.neighbors(v))
#            
#        yield v
#    
#
##        desired_nodes = len(nodes_selection)        
##        
##        if(desired_links < 0):
##            desired_links = int(0.41*desired_nodes**1.57)
#            
#        return ig_graph
#        
        
    


            
        
            
        
            
        
        