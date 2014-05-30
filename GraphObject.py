# -*- coding: utf-8 -*-
"""
Created on Sat Mar 08 16:06:30 2014

@author: t-cflore
"""
import abc
import igraph as ig
import networkx as nx
import numpy as np

class GraphObject(object):
    
    __metaclass__ = abc.ABCMeta 
    
    _IGRAPH = 1
    _NETWORKX = 2
    
    @staticmethod
    def __init__(self,graph_object):
        '''Call the appropiate constructor depending on the type of graph object'''
        if type(graph_object) is ig.Graph:
            return IgGraph(graph_object)
        else:
            return NxGraph(graph_object)
    
    @abc.abstractmethod
    def GetNodesAttribute(self,attribute_name,map_function=lambda x: x):
        '''Returns a dictionary with keys belonging to node id's and values to
        the attribute passed as argument'''
        
    @abc.abstractmethod
    def GetEdgesAttribute(self,attribute_name,map_function=lambda x: x):
        '''Returns a dictionary with keys belonging to edges id's and values to
        the attribute passed as argument'''
        
    @abc.abstractmethod
    def GetGiantComponent(self):
        '''Return the weakly giant component of a graph'''
        
    @abc.abstractmethod
    def SubGraphByNodes(self,nodes):
        '''Return a subgraph using only the selected nodes'''
        
    @abc.abstractmethod
    def SubGraphByEdges(self,edges):
        '''Return a subgraph using only the selected edges'''
    
    @abc.abstractmethod
    def NumberOfNodes(self):
        '''Return the number of nodes in a graph'''
        
    @abc.abstractmethod
    def NumberOfEdges(self):
        '''Return the number of nodes in a graph'''
    
class NxGraph(GraphObject):
    
    def __init__(self,graph_object):

        self._graph = graph_object
        
    def GetNodesAttribute(self,attribute_name,map_function=lambda x: x):
        '''Returns a dictionary with keys belonging to node id's and values to
        the attribute passed as argument'''
        gx = self._graph
        return {v:map_function(gx.node[v][attribute_name]) for v in gx.node.keys()}
        
    def GetEdgesAttribute(self,attribute_name,map_function=lambda x: x):
        '''Returns a dictionary with keys belonging to edges id's and values to
        the attribute passed as argument'''
        gx = self._graph
        return {(x,y):map_function(gx.edge[x][y]['BiomassIngested']) for x in gx.edge for y in gx.edge[x]}
        
    def GetGiantComponent(self):
        '''Return the weakly giant component of a graph'''
        gx = self._graph
        return NxGraph(nx.weakly_connected_component_subgraphs(gx)[0])

    def SubGraphByNodes(self,nodes):
        '''Return a subgraph using only the selected nodes'''
        return NxGraph(self._graph.subgraph(nodes))
        
    def SubGraphByEdges(self,edges):
        '''Return a subgraph using only the selected edges'''
        gx = self._graph.copy()
        edges_to_remove = [o for o in gx.edges() if o not in edges]        
        gx.remove_edges_from(edges_to_remove)
        return NxGraph(gx) 


    def NumberOfNodes(self):
        '''Return the number of nodes in a graph'''
        return self._graph.number_of_nodes()
    
    def NumberOfEdges(self):
        '''Return the number of nodes in a graph'''
        return self._graph.number_of_edges()
        
class IgGraph(GraphObject):
    
    def __init__(self,graph_object):

        self._graph = graph_object
        
    def GetNodesAttribute(self,attribute_name,map_function=lambda x: x):
        '''Returns a dictionary with keys belonging to node id's and values to
        the attribute passed as argument'''
        gi = self._graph
        return {v.index:map_function(v[attribute_name]) for v in gi.vs}
        
    def GetEdgesAttribute(self,attribute_name,map_function=lambda x: x):
        '''Returns a dictionary with keys belonging to edges id's and values to
        the attribute passed as argument'''
        gi = self._graph
        return {e.index:map_function(e[attribute_name]) for e in gi.es}

    def GetGiantComponent(self):
        '''Return the weakly giant component of a graph'''
        xmax = 0        
        gi = self._graph
        vec = []
        for x in gi.components(mode=1):
            if len(x) > xmax:
                xmax = len(x)
                vec = x
        return IgGraph(gi.subgraph(vec))
        
    def SubGraphByNodes(self,nodes):
        '''Return a subgraph using only the selected nodes'''
        return IgGraph(self._graph.subgraph(nodes))
        
    def SubGraphByEdges(self,edges):
        '''Return a subgraph using only the selected edges'''
        return IgGraph(self._graph.subgraph_edges(edges))
        
    def NumberOfNodes(self):
        '''Return the number of nodes in a graph'''
        return self._graph.vcount()
        
    def NumberOfEdges(self):
        '''Return the number of nodes in a graph'''
        return self._graph.ecount()
