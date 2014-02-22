# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:32:13 2014

@author: t-cflore
"""

from NetworkXCreator import NetworkXCreator
import networkx as nx
import numpy as np
import sys as sys
import matplotlib.pyplot as plt


class BasicExperiments:
    
    @staticmethod
    def GeneralProperties():
        """
        
        """
        
        graphs_small = NetworkXCreator.GET_NETX_TIME_LAYERS('data/small',name='unmerged')
        graphs_big = []        
        #graphs_big = NetworkXCreator.GET_NETX_TIME_LAYERS('data/big_sel',time_layers=[1200,1225,1250])
        
        return graphs_small,graphs_big
        
        
    @staticmethod
    def RUN_BASIC_EXPERIMENTS(networks,n_trials = 1):
        
        for g in networks:
            n_edges = g.number_of_edges()
            n_nodes = g.number_of_nodes()
            
            cl_random = []
            d_random = []
            print 'Edges = %i, Nodes = %i' % (n_edges, n_nodes)
            for trial in range(n_trials):
                for i in range(10): 
                    g_rand = nx.gnm_random_graph(n_nodes,n_edges)   
                    if (nx.is_connected(g_rand)==True): 
                        print 'I am connected'
                        sys.stdout.flush()
                        break
                
                cl_random.append(nx.average_clustering(g_rand))
                d_random.append(nx.average_shortest_path_length(g_rand))
            
            clrand = np.mean(cl_random)
            dran = np.mean(d_random)
            BasicExperiments.PRINT_GENERAL_PROP_LINE(g,clrand,dran)
            


    @staticmethod
    def PRINT_GENERAL_PROP_LINE(G,clran,dran):
        S = 1.0*G.number_of_nodes()
        L = 1.0*G.number_of_edges()
        C = L/(S*S)
        LS = L/S
        Cl = nx.average_clustering(G)
        D = nx.average_shortest_path_length(G)
        
        print('%i %i %.4f %.4f %.4f %.4f %.4f %.4f %.4f ' % (G.number_of_nodes(),G.number_of_edges(),C,LS,Cl,clran, Cl/clran,D,dran))
        
    @staticmethod    
    def my_cdf(values):

        values = np.array(values)
        values.sort()
        
        elements = np.unique(values)
        elements = elements[::-1]
    
        counting = np.zeros([len(elements),1])
        
        for i in range(len(elements)):
            counting[i] = len(values[values==elements[i]])
        
        cdf = np.cumsum(1.0*counting)/np.sum(counting)
        
        return elements[::-1],cdf[::-1]
    
    @staticmethod
    def my_pdf(values):
    
        values = np.array(values)
        values.sort()
        
        elements = np.unique(values)
        elements = elements[::-1]
    
        counting = np.zeros([len(elements),1])
        
        for i in range(len(elements)):
            counting[i] = len(values[values==elements[i]])
        
        pdf = 1.0*counting/np.sum(counting)
        
        return elements[::-1],pdf[::-1]
    
    
    @staticmethod
    def u_plot_degree_CDF(G):
        
        degrees = G.degree().values()
        
        elems,cdf = BasicExperiments.my_cdf(degrees)
           
        plt.loglog(elems, cdf,'yo')
        #plt.title('N = %i E= % i')
        plt.xlabel('# of trophic links')
        plt.ylabel('CDF value')
        
        
        plt.show()
        
    @staticmethod
    def PLOT_ALL_CDF(graphs, nx, ny):
        n_nets = len(graphs)        
        
        plt.figure(1)
        
        t = 1200
        for i in range(n_nets):
            plt.subplot(ny,nx,i+1)
            
            degrees = graphs[i].degree().values()
            S = graphs[i].number_of_nodes()
            L = graphs[i].number_of_edges()
            
            elems,cdf = BasicExperiments.my_cdf(degrees)
             
            plt.cla()
            plt.loglog(elems, cdf,'yo')
            plt.title('S = %i L = %i' % (S,L))
            
         
            if(i>7): 
                plt.xlabel('# of trophic links',fontsize=12)
            else:
                ga = plt.gca()
                ga.axes.set_xticklabels([])                
            
            if(i % 4 == 0):
                plt.ylabel('CDF',fontsize=12)
            else:
                ga = plt.gca()
                ga.axes.set_yticklabels([])
                
            plt.ylim(ymin=10**(-4))
            plt.ylim(ymax=1)
            t += 1
        plt.show()
     
    @staticmethod       
    def PLOT_ALL_DEGREE_INFORMATION(G):
        

        plt.figure(1)
        plt.cla()
        for i in range(3):
            if(i == 0): 
                degrees = G.degree().values()
                title = 'All-degree'
            if(i == 1): 
                degrees = G.in_degree().values()
                title = 'In-degree'
            if(i == 2): 
                degrees = G.out_degree().values()
                title = 'Out-degree'
            
            elems,cdf = BasicExperiments.my_cdf(degrees)
            elems,pdf = BasicExperiments.my_pdf(degrees)
            
            plt.subplot(2,3,i+1)
            plt.cla()
            plt.loglog(elems, pdf,'yo')
            if(i==0):
                plt.ylabel('PDF',fontsize=12)
            plt.title(title, fontsize=16)
            
            plt.subplot(2,3,i+4)
            plt.cla()
            plt.loglog(elems, cdf,'yo')
            if(i==0):
                plt.ylabel('CDF',fontsize=12)
            #plt.title('S = %i L = %i' % (S,L))
            plt.xlabel('# trophic links',fontsize=12)
            
            
            
    
    @staticmethod
    def u_plot_degree_PDF(G,title_label):
        
        degrees = G.degree().values()
        
        elems,pdf = BasicExperiments.my_pdf(degrees)
           
        plt.loglog(elems, pdf,'yo')
        plt.title('Probability Distribution for N ~ 1000')
        plt.xlabel('# of trophic links')
        plt.ylabel('CDF value')
        
        
        plt.show()    
        
