from multiprocessing import Pool
from time import sleep
from random import randint
import os
from code import *
from time import time
 
 
class AsyncFactory:
    def __init__(self, func):
        self.func = func
        #self.cb_func = cb_func
        self.pool = Pool()
        self.results = []
        self.map_results = []
 
    def call(self,*args, **kwargs):
        #print args
        self.pool.apply_async(self.func, args, kwargs, callback=self.results.append)
        
    def amap(self,iterable):
        self.map_results = self.pool.map(self.func,iterable)
 
    def wait(self):
        self.pool.close()
        self.pool.join()

def iter_data(iterable):
    x,y = DataManipulation.GET_DATASETS(*iterable)
    return x,y

def bigdata(*args,**kwargs):
    print args
    print kwargs  
    print ' hola '
    x,y = DataManipulation.GET_DATASETS(*args,**kwargs)
    return [x,y]
 
def square(x):
    sleep_duration = randint(1,5)
    print "PID: %d \t Value: %d \t Sleep: %d" % (os.getpid(), x ,sleep_duration)
    sleep(sleep_duration)
    return x*x
 

if __name__=='__main__':
    nodes_dir='data/nodes/'
    edges_dir='data/edges/'
    time_step=1223
    
    node_files = [ nodes_dir+f for f in listdir(nodes_dir) if isfile(join(nodes_dir,f)) ]
    edge_files = [ edges_dir+f for f in listdir(edges_dir) if isfile(join(edges_dir,f)) ]
    
    async_square = AsyncFactory(iter_data)
    t = time()
    async_square.amap(zip(node_files,edge_files,len(node_files)*[1223]))
    print time()-t
    #for i in range(10):
     #   async_square.call(i)
    #t = time()
    #for f_node,f_edges in zip(node_files, edge_files):       
    #    async_square.call(nodes_dir+f_node,edges_dir+f_edges,time_step=1223)
    # 
    #async_square.wait()
    #print time()-t