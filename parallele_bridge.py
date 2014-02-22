from multiprocessing import Pool
from time import sleep
from random import randint
import os
from code import *
from time import time


class ParallelProccessFactory:
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