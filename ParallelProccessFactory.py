from multiprocessing import Pool
from time import sleep
from random import randint
import os
#from code import *
from time import time


class ParallelProccessFactory:
    def __init__(self, func):
        '''A small factory to avoid the pikcle serialization problem during multiprocessing.
           Use this class at your own risk
        '''
        self.func = func
        #self.cb_func = cb_func
        self.pool = Pool()
        self.async_results = []
        self.map_results = []
 
    def call(self,*args, **kwargs):
        '''Function to use for asyncronous calls, it is useful for fuctions that have 
        both normal and optional arguments'''
        #print args
        self.pool.apply_async(self.func, args, kwargs, callback=self.async_results.append)
        
    def pmap(self,iterable):
        ''' a map funtion that returns all the results in the exact order'''
        self.map_results = self.pool.map(self.func,iterable)
 
    def wait(self):
        '''Need to call this function after realizing all your calls for getting all your results'''
        self.pool.close()
        self.pool.join()