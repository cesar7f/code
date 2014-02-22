# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:54:10 2014

@author: t-cflore
"""

#%% 
# PROFILING LXML DF_TO_GEXF_FUNCTION
execfile('code/DataManipulation.py')
execfile('code/KMeansAggregation.py')
#nodes,edges = MetaAnalysis.GET_BIG_DICTIONARY()
#DataManipulation.DF_TO_GEPPHI_CSV(nodes[0.0],edges[0.0],file_nodes='nodes_0.00.csv',file_edges='edges_0.00.csv')
#df_nodes,df_edges = DataManipulation.GET_DATASETS(file_nodes = 'data/nodes/nodes_0.50_.txt', file_edges = 'data/edges/edges_0.50_.txt',time_step=1223)
#DataManipulation.DF_TO_GEXF(df_nodes, df_edges,'real_thing.gexf')
#nodes
nodes,edges = MetaAnalysis.GET_BIG_DICTIONARY()

#%%
# Creating the function MetaAnalysis.DF_TO_GML()
execfile('code/DataManipulation.py')
DataManipulation.DF_TO_GRAPHML(nodes[0.0],edges[0.0],file_output='test2.xml')

#%%
# Random experiments with python
import igraph as ig
%timeit g_ig = ig.Graph.Read_GraphML('test2.xml')
#ig.plot(g_ig)    

#%% Testing parallel things
    
from IPython.parallel import Client

c = Client()
print c.ids
print c[:].apply_sync(lambda : "Hello, World")

dview = c[:]

@dview.remote(block=True)
def getpid():
    import os
    return os.getpid()
    
getpid()

#%%
def return_dic(num,strs):
    return num,strs
    
aa =map(return_dic,[1,2,3,4],['a','b','c','d'])

#%%
from IPython.kernel import client as cl

mec = Client.MultiEngineClient()

#%%
import math
from multiprocessing import Pool

if __name__ == '__main__':
    pool = Pool(processes=2)
    print pool.map(math.sqrt, [1,4,9,16])
    
#%%
from os import listdir
from os.path import isfile, join
from IPython.parallel import Client
from time import time
execfile('code/DataManipulation.py')
nodes_dir='C:/Users/t-cflore/Dropbox/microsoft_research/data/nodes/'
edges_dir='C:/Users/t-cflore/Dropbox/microsoft_research/data/edges/'
time_step=1223


node_files = [ f for f in listdir(nodes_dir) if isfile(join(nodes_dir,f)) ]
edge_files = [ f for f in listdir(edges_dir) if isfile(join(edges_dir,f)) ]
        
node_files.sort()
edge_files.sort()

assert(len(node_files) == len(edge_files))        

nodes = dict()
edges = dict()

for i in range(len(node_files)):
    node_files[i] = nodes_dir + node_files[i]
    edge_files[i] = edges_dir + edge_files[i]



rc = Client()
dview = rc[:] 
dview.push({'time_step':time_step,'GET_DATASETS':DataManipulation.GET_DATASETS})
#dview.push(time_step)
#dview.push(DataManipulation.GET_DATASETS)

@dview.parallel(block=True)
def get_parallel_dataset(file_node,file_edge):
    
    df_nodes,df_edges = GET_DATASETS(file_nodes=file_node,file_edges=file_edge,time_step=time_step,drop_sources=True)
    return file_node,file_edge

t = time()
get_parallel_dataset.map(node_files,edge_files)
print t-time()


    

#%%

# Try first by parallelizing the creation of data sets
try:        
    rc = Client()
    dview = rc[:] 
    get_parallel_dataset.map(node_files,edge_files)
    
    @dview.parallel(block=True)
    def get_parallel_dataset(file_node,file_edge):
        #df_nodes,df_edges = DataManipulation.GET_DATASETS(file_nodes=file_node,file_edges=file_edge,time_step=time_step,drop_sources=True)
        #return df_nodes,df_edges
        print time_step
        return 1,3
    
except: #Did not work, -> extract sequentially
    print 'exception found, trying next thing'
    for f_node,f_edges in zip(node_files, edge_files):
        node_key = float(f_node.split('_')[1])
        edge_key = float(f_edges.split('_')[1])
        nodes[node_key],edges[edge_key] = DataManipulation.GET_DATASETS(file_nodes=f_node, file_edges=f_edges,time_step=time_step,drop_sources=True)
#%%
run testing_pool.py

#%%
def fa(c,d,a=3,b=5):
    print c,d
    print a,b
    print a
    
def fs(args,kwargs):
    fa(*args,**kwargs)
    
def fd(*args, **kwargs):
    print args
    fa(*args,**kwargs)
    
def fr(**kwargs):
    fa(3,4,**kwargs)

def fmain(f,args)

fs( (1,2),{'a':4} )
fd( 1,2 ,a=7)

fr(b=7)

#%%
from multiprocessing import Pool

def f(x, *args, **kwargs):
    print x, args, kwargs

args, kw = (1,2,3), {'cat': 'dog'}

print "# Normal call"
f(0, *args, **kw)

print "# Multicall"
P = Pool()
sol = [P.apply_async(f, (x,) + args, kw) for x in range(2)]
P.close()
P.join()

for s in sol: s.get():
    
#%%
def ff(a,b,c):
    print a,b,c
    
def ss(a,*args):
    print a,args
    ff(*args)
    
ss(1,2,3,4)