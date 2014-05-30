# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:38:57 2014

@author: t-cflore
"""

import numpy as np
#from NetworkXCreator import NetworkXCreator
import networkx as nx
from DataManipulation import DataManipulation
from scipy.cluster.vq import kmeans,vq,whiten
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pdb
from NetworkSampling import NetworkSampling
import prettyplotlib as ppl
import igraph as ig
def pcolormesh(*args, **kwargs):
    from prettyplotlib.colors import blue_red, blues_r, reds   
    from prettyplotlib.utils import remove_chartjunk, maybe_get_fig_ax
    
    """
    Use for large datasets

    Non-traditional `pcolormesh` kwargs are:
    - xticklabels, which will put x tick labels exactly in the center of the
    heatmap block
    - yticklables, which will put y tick labels exactly aligned in the center
     of the heatmap block
     - xticklabels_rotation, which can be either 'horizontal' or 'vertical'
     depending on how you want the xticklabels rotated. The default is
     'horizontal', but if you have xticklabels that are longer, you may want
     to do 'vertical' so they don't overlap.
     - yticklabels_rotation, which can also be either 'horizontal' or
     'vertical'. The default is 'horizontal' and in most cases,
     that's what you'll want to stick with. But the option is there if you
     want.
    - center_value, which will be the centered value for a divergent
    colormap, for example if you have data above and below zero, but you want
    the white part of the colormap to be equal to 10 rather than 0,
    then specify 'center_value=10'.
    """
    # Deal with arguments in kwargs that should be there, or need to be taken
    #  out
    fig, ax, args, kwargs = maybe_get_fig_ax(*args, **kwargs)

    x = args[0]

    kwargs.setdefault('vmax', x.max())
    kwargs.setdefault('vmin', x.min())

    center_value = kwargs.pop('center_value', 0)

    # If
    divergent_data = False
    if kwargs['vmax'] > 0 and kwargs['vmin'] < 0:
        divergent_data = True
        kwargs['vmax'] += center_value
        kwargs['vmin'] += center_value

    # If we have both negative and positive values, use a divergent colormap
    if 'cmap' not in kwargs:
        # Check if this is divergent
        if divergent_data:
            kwargs['cmap'] = blue_red
        elif kwargs['vmax'] <= 0:
                kwargs['cmap'] = blues_r
        elif kwargs['vmax'] > 0:
            kwargs['cmap'] = reds

    if 'xticklabels' in kwargs:
        xticklabels = kwargs['xticklabels']
        kwargs.pop('xticklabels')
    else:
        xticklabels = None
    if 'yticklabels' in kwargs:
        yticklabels = kwargs['yticklabels']
        kwargs.pop('yticklabels')
    else:
        yticklabels = None

    if 'xticklabels_rotation' in kwargs:
        xticklabels_rotation = kwargs['xticklabels_rotation']
        kwargs.pop('xticklabels_rotation')
    else:
        xticklabels_rotation = 'horizontal'
    if 'yticklabels_rotation' in kwargs:
        yticklabels_rotation = kwargs['yticklabels_rotation']
        kwargs.pop('yticklabels_rotation')
    else:
        yticklabels_rotation = 'horizontal'

    
    
    use_colorbar = kwargs.pop('use_colorbar',True)
    ax_colorbar = kwargs.pop('ax_colorbar', None)
    orientation_colorbar = kwargs.pop('orientation_colorbar', 'vertical')

    p = ax.pcolormesh(*args, **kwargs)
    ax.set_ylim(0, x.shape[0])

    # Get rid of ALL axes
    remove_chartjunk(ax, ['top', 'right', 'left', 'bottom'])

    if xticklabels:
        xticks = np.arange(0.5, x.shape[1] + 0.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=xticklabels_rotation)
    if yticklabels:
        yticks = np.arange(0.5, x.shape[0] + 0.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, rotation=yticklabels_rotation)

    # Show the scale of the colorbar
    fig.colorbar(p, ax=ax_colorbar, use_gridspec=True,
                orientation=orientation_colorbar)
    return fig


class GeneralPlotter:

    _FG_RGB_COLORS = {10:[0,100,0],
                      11:[255,0,0],
                      12:[0,0,100],
                      13:[180,255,180],
                      14:[255,180,180],
                      15:[180,180,255],
                      16:[0,255,0],
                      17:[100,0,0],
                      18:[0,0,255],
                      100:[0,0,0]} # Source Functional group

    @staticmethod
    def PLOT_ADULT_VS_JUVENILE_MASS(df_nodes,fontaxis=14):
        my_colors = []
        df_nodes = df_nodes[df_nodes.FunctionalGroup <= 20]
        for fg in df_nodes.FunctionalGroup:
            my_colors.append(np.array(GeneralPlotter._FG_RGB_COLORS[fg])/255.0)
            
        ssmin = 10
        ssmax = 80
        ss = df_nodes.CohortAbundance.values*df_nodes.IndividualBodyMass.values
        ss = ssmin + (ss-np.min(ss))*(ssmax-ssmin)/(np.max(ss)-np.min(ss))
         
        plt.scatter(np.log10(df_nodes.AdultMass),np.log10(df_nodes.JuvenileMass),c=my_colors,s=ss, alpha=0.6)
        plt.xlabel('log10 (Adult Mass)',fontsize=fontaxis)
        plt.ylabel('log10 (Juvenile Mass)',fontsize=fontaxis)
       
    @staticmethod
    def PLOT_Y_VS_X(df_nodes,yvar='JuvenileMass',xvar='AdultMass',fontaxis=14):
        my_colors = []
        df_nodes = df_nodes[df_nodes.FunctionalGroup <= 20]
        for fg in df_nodes.FunctionalGroup:
            my_colors.append(np.array(GeneralPlotter._FG_RGB_COLORS[fg])/255.0)
            
        ssmin = 10
        ssmax = 80
        ss = df_nodes.CohortAbundance.values*df_nodes.IndividualBodyMass.values
        ss = ssmin + (ss-np.min(ss))*(ssmax-ssmin)/(np.max(ss)-np.min(ss))
         
        plt.scatter(np.log10(df_nodes[xvar]), np.log10(df_nodes[yvar]),c=my_colors,s=ss, alpha=0.6)
        plt.xlabel('log10 (%s)' % xvar, fontsize=fontaxis)
        plt.ylabel('log10 (%s)' % yvar, fontsize=fontaxis)
        
        
    @staticmethod
    def PLOT_PRED_VS_PREY(df_nodes,df_edges,field_name = 'IndividualBodyMass',log_filter=-1000):
        df_nodes = df_nodes[df_nodes.FunctionalGroup <= 30]
        df_nodes['Biomass'] = df_nodes['IndividualBodyMass']*df_nodes['CohortAbundance']
        df_edges = df_edges[df_edges.Pred_ID >= 0]
        df_edges = df_edges[df_edges.Prey_ID >= 0]
        df_edges[df_edges['Prey_ID'].isin(df_nodes.ID.unique())]
        df_edges[df_edges['Pred_ID'].isin(df_nodes.ID.unique())]
        df_edges = df_edges[np.log10(df_edges.BiomassIngested) > log_filter]
        df_edges = df_edges[['Prey_ID','Pred_ID']]
        
        dfni = df_nodes.copy()
        dfni = dfni.set_index('ID',drop=False)
        
        df_mass = df_edges.join(dfni, on='Prey_ID')
        df_mass = df_mass.rename(columns = {field_name : 'mass_prey'})
        df_mass = df_mass[['Prey_ID','Pred_ID','mass_prey']]
        df_mass = df_mass.join(dfni, on='Pred_ID')
        df_mass = df_mass.rename(columns = {field_name : 'mass_pred'})
        df_mass = df_mass[['Pred_ID','Prey_ID','mass_pred','mass_prey']]
        
    
        #xx = [df_mass['mass_pred'].min(),df_mass['mass_pred'].max()]
        #yy = [df_mass['mass_pred'].min(),df_mass['mass_pred'].max()]
              
        plt.loglog(df_mass['mass_pred'],df_mass['mass_prey'],'ro')
        plt.loglog([0.01,1e7],[0.01,1e7])
        
        plt.xlabel('Predator %s' % field_name,fontsize=16)
        plt.ylabel('Prey %s' % field_name,fontsize=16)
                  
           
           
    @staticmethod
    def MASS_RATIO_DISTANCE(u,v):
        '''Similarity measure based in mass ratio. Similar to what Mandigley is
        doing for the mergint process''' 
        if u[2]==v[2]:
            diff_vec = 1.0*np.abs(u[:2]-v[:2])/np.minimum(u[:2],v[:2])
            return np.sqrt((diff_vec**2).sum())
        else:
            return np.inf
        
        
                
    @staticmethod
    def IMPROVED_K_MEANS(df_nodes,df_edges, k=80):
        df_nc = df_nodes.copy()
        
        
        k=90
        data = df_nc[['JuvenileMass', 'AdultMass']]
        data = data.as_matrix()
        
        data = np.log10(data)
        data = whiten(data)
        data = np.c_[data,1000.*df_nodes.FunctionalGroup.values]
        
        centroids,_ = kmeans(data,k)
        idx,_ = vq(data,centroids)
        
        df_nc['cluster'] = pd.Series(idx, index=df_nc.index)
        
        x = data[:,0]
        y = data[:,1]
        c_clust = df_nc['cluster']
        c_func = df_nc['FunctionalGroup']
        
        
        plt.figure(1)
        plt.clf()
        plt.subplot(1,3,1)
        plt.scatter(x, y, c=c_func)
        plt.subplot(1,3,2)
        plt.scatter(x, y, c=c_clust)
        plt.subplot(1,3,3)
        plt.scatter(centroids[:,0], centroids[:,1],c=range(np.max(c_clust)+1))
        
        df_nodes['cluster']=-1
        df_nodes.loc[df_nc.index,'cluster']=df_nc['cluster']
        
        for i in df_nodes['FunctionalGroup'].unique():
            for j in df_nodes['FunctionalGroup'].unique():
                if j <= i: continue
                if len(numpy.intersect1d(aa.loc[i]['cluster'].unique(),aa.loc[j]['cluster'].unique()))>0:
                    print 'Something bad happened'
        
        n_links = 3
        edges = []
        maxc = np.max(idx)   
        for c1 in range(maxc):
            for c2 in range(maxc+1):
                nodes_a = df_nodes[df_nodes.cluster==c1]['ID'].values
                nodes_b = df_nodes[df_nodes.cluster==c2]['ID'].values
                
                if (len(df_edges[(df_edges.Pred_ID.isin(nodes_a))&(df_edges.Prey_ID.isin(nodes_b))].index) >= n_links):
                    edges.append((c1,c2))
                    continue
                if (len(df_edges[(df_edges.Pred_ID.isin(nodes_b))&(df_edges.Prey_ID.isin(nodes_a))].index) >= n_links):
                    edges.append((c2,c1))
                    continue        
                    
        #G = nx.DiGraph()
        #G.add_edges_from(edges)
                
        #Ga = NetworkXCreator.DF_TO_NETX(df_nodes,df_edges)
        #G = nx.connected_component_subgraphs(Ga)[0]     
        #G = nx.connected_component_subgraphs(G)[0]
        
        #FilterData.PRINT_BASIC_INFORMATION(G)
        #nx.write_gexf(G,'my_hope.gexf')

                
                
                    
        
        
    @staticmethod
    def PLOT_PREY_VS_PRED_MASS(df_nodes, df_edges, time_step=-1, f_group=-1):
        
        if(time_step > -1):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            df_edges = df_edges[df_edges.time_step==time_step]
            
        if f_group > -1: 
            df_nodes = df_nodes[df_nodes['FunctionalGroup']==f_group]
        
        df_edges = df_edges[df_edges.Prey_ID.isin(df_nodes.ID.unique())]
        
        xx = [df_nodes.IndividualBodyMass[df_nodes.ID==x].values[0] for x in df_edges.Pred_ID]
        yy = [df_nodes.IndividualBodyMass[df_nodes.ID==x].values[0] for x in df_edges.Prey_ID]
        #xx = df_nodes.IndividualBodyMass[df_edges.Pred_ID]
        #yy = df_nodes.IndividualBodyMass[df_edges.Prey_ID]
        #df_nodes.reindex(columns=['ID'])
        
        #xx = df_nodes.ix[df_edges['Pred_ID']]['IndividualBodyMass']
        #yy = df_nodes.ix[df_edges['Prey_ID']]['IndividualBodyMass']


        plt.figure(1)
        plt.clf()
        plt.loglog(xx, yy,'bo')


    @staticmethod
    def PLOT_DENSITY_MASS(df_nodes,time_step=-1,f_group=-1):

        if(time_step > -1):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            #df_edges = df_edges[df_edges.time_step==time_step]
            
        if f_group > -1: df_nodes = df_nodes[df_nodes['FunctionalGroup']==f_group]
        
        x = df_nodes['AdultMass']   
        y = df_nodes['JuvenileMass']
        #y = df_nodes['CohortAbundance']
        
        class_id = df_nodes['FunctionalGroup']
        
        plt.figure(1)
        plt.clf()
        plt.subplot(1,2,1)
        plt.scatter(x, y, c=class_id)
        plt.subplot(1,2,2)
        plt.scatter(np.log10(x), np.log10(y), c=class_id)
        #plt.legend(tuple([str(label) for label in class_id.unique()]))

        
    @staticmethod
    def PLOT_SORTED_MATRIX(df_nodes=None,df_edges=None,matrix=None,igraph_object=None,axis=0):
        '''Plot the predation matrix'''        
        
        if igraph_object:
            matrix = np.asarray(igraph_object.get_adjacency().data)
        elif not matrix:
            igraph_object = DataManipulation.DF_TO_IGRAPH(df_nodes,df_edges)
            matrix = np.asarray(igraph_object.get_adjacency().data)

        sum_rows = matrix.sum(axis=axis)
        new_idx = np.argsort(sum_rows)
        
        plt.imshow(matrix.take(new_idx,axis=0).take(new_idx,axis=1))

    @staticmethod
    def PLOT_MATRIX(matrix,idx=None,axis=0,sort = True, show_bar=True,new_figure=False):
        matrix = np.asarray(matrix)        
        if idx is None:
            if sort == True:            
                sum_rows = matrix.sum(axis=axis)            
                idx = np.argsort(sum_rows)
            else:
                idx = range(matrix.shape[0])
        
        if new_figure:
            plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        
        
        plt.imshow(matrix.take(idx,axis=0).take(idx,axis=1),interpolation='nearest')
        if show_bar:
            plt.colorbar()
    
    @staticmethod
    def SORT_MATRIX(matrix,idx=None,axis=0):
        matrix = np.asarray(matrix)        
        if idx is None:
            sum_rows = matrix.sum(axis=axis)            
            idx = np.argsort(sum_rows)
            
        return matrix.take(idx,axis=0).take(idx,axis=1)
        
    
    @staticmethod
    def K_MEANS_NODE_AGG(df_nodes,df_edges,k=80,n_links=2,time_step=-1):
        """Aggregate the data by kmeans thresshold"""
        G = nx.Graph()
        if(time_step > -1):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            df_edges = df_edges[df_edges.time_step==time_step]
        
        data = df_nodes[['JuvenileMass', 'AdultMass']]
        data = data.as_matrix()
        data = np.log10(data)        
        
        centroids,_ = kmeans(data,k)
        idx,_ = vq(data,centroids)
        
        df_nodes['cluster'] = pd.Series(idx, index=df_nodes.index)
        
        edges = []
        maxc = np.max(idx)        
        
        for c1 in range(maxc):
            for c2 in range(maxc+1):
                nodes_a = df_nodes[df_nodes.cluster==c1]['ID'].values
                nodes_b = df_nodes[df_nodes.cluster==c2]['ID'].values
                
                if (len(df_edges[(df_edges.Pred_ID.isin(nodes_a))&(df_edges.Prey_ID.isin(nodes_b))].index) >= n_links):
                    edges.append((c1,c2))
                    continue
                if (len(df_edges[(df_edges.Pred_ID.isin(nodes_b))&(df_edges.Prey_ID.isin(nodes_a))].index) >= n_links):
                    edges.append((c1,c2))
                    continue
                
                
       
        G.add_edges_from(edges)
        
        #Ga = NetworkXCreator.DF_TO_NETX(df_nodes,df_edges)
        #G = nx.connected_component_subgraphs(Ga)[0]     
        G = nx.connected_component_subgraphs(G)[0]
        
        FilterData.PRINT_BASIC_INFORMATION(G)
        
    
    @staticmethod
    def PLOT_CDF(degrees):
        
        values = np.array(degrees)
        values.sort()
        
        elements = np.unique(values)
        elements = elements[::-1]
    
        counting = np.zeros([len(elements),1])
        
        for i in range(len(elements)):
            counting[i] = len(values[values==elements[i]])
        
        cdf = np.cumsum(1.0*counting)/np.sum(counting)
        
        
           
        plt.semilogy(elements[::-1], cdf[::-1],'yo')
        #plt.title('N = %i E= % i')
        plt.xlabel('# of trophic links')
        plt.ylabel('CDF value')
        
        
        plt.show()

    @staticmethod
    def PLOT_RELATIVE_ABUNDANCE(ab,nsampling = 200):
            n = len(ab)
            selection=np.random.choice(range(n),p=NetworkSampling.FITNESS_PROBABILITY(ab),size=nsampling)
            uniq_keys = np.unique(selection)
            bins = uniq_keys.searchsorted(selection)
            plt.hist(np.log(np.bincount(bins)),bins=10)

    @staticmethod
    def PLOT_MATRIX_SAMPLING(matrix,idx=None,sampling_size=None,colorm='jet'):
        from prettyplotlib import brewer2mpl
        import pylab
        red_purple = brewer2mpl.get_map('Blues', 'Sequential', 9).mpl_colormap
        green_purple = brewer2mpl.get_map('PRGn', 'diverging', 11).mpl_colormap
        if sampling_size == None:
            sampling_size = [1e0,1e1,1e2,1e3,1e4,1e300]
        
        fig,ax = ppl.subplots(nrows=2,ncols=3)
        for ss in range(len(sampling_size)):
            nn = sampling_size[ss]
            pmatrix = 1 - np.exp(-nn*matrix)
            aa = ax[ss/3,ss%3]
            
            #plt.subplot(2,3,ss+1)
            #GeneralPlotter.PLOT_MATRIX(pmatrix,axis=1)
            #print ss/3,ss%3
            pmatrix = GeneralPlotter.SORT_MATRIX(pmatrix,idx=idx,axis=1) 
            pmatrix = np.flipud(pmatrix)
            #im = aa.matshow(pmatrix, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
            #pp=aa.imshow(pmatrix,interpolation='nearest',cmap=colorm)    
            #fig.colorbar(pp,ax=aa)
            pcolormesh(fig,aa,pmatrix,center_value=0.5, ax_colorbar=aa,cmap=red_purple)
            aa.set_xticks([])
            aa.set_yticks([])
            
            aa.set_title(r'$n = %i$' % sampling_size[ss],fontsize=20)
            if ss==5:
                plt.title('$n = \infty$',fontsize=20)

      
      
      
    @staticmethod
    def PLOT_CDF_SAMPLING(matrix, sampling_size = None):
        
        if sampling_size == None:
            sampling_size = [1e0,1e1,1e2,1e3,1e4,1e300]
        
        fig,ax = ppl.subplots(nrows=2,ncols=3)
        for ss in range(len(sampling_size)):
            nn = sampling_size[ss]
            pmatrix = 1 - np.exp(-nn*matrix)            
            
            #degrees = np.sum(pmatrix>0.8,axis=0) + np.sum(pmatrix>0.999,axis=1)
            degrees = np.round(np.sum(pmatrix,axis=0) + np.sum(pmatrix,axis=1))
            
        
            aa = ax[ss/3,ss%3]
            
            values = np.array(degrees)
            values.sort()
            
            elements = np.unique(values)
            elements = elements[::-1]
        
            counting = np.zeros([len(elements),1])
            
            for i in range(len(elements)):
                counting[i] = len(values[values==elements[i]])
            
            cdf = np.cumsum(1.0*counting)/np.sum(counting)
            
            
               
            aa.semilogy(elements[::-1], cdf[::-1],'bo')
            #plt.title('N = %i E= % i')
            aa.set_xlabel('# of trophic links')
            aa.set_ylabel('CDF value')
            
            aa.set_title(r'$n = %i$' % sampling_size[ss],fontsize=20)
            if ss==5:
                plt.title('$n = \infty$',fontsize=20)
            
            
    @staticmethod
    def PLOT_SAMPLING_BASIC_METRICS(matrix, sampling_size = None):
        
        if sampling_size == None:
            sampling_size = np.arange(1,6000,1)
        
        L = np.zeros(len(sampling_size))
        S = np.zeros(len(sampling_size))
        C = np.zeros(len(sampling_size))
        Lrand = np.zeros(len(sampling_size))
        Crand = np.zeros(len(sampling_size))
        
        for ss in range(len(sampling_size)):
            nn = sampling_size[ss]
            pmatrix = 1 - np.exp(-nn*matrix)            
            ns = pmatrix.shape[0]
            adjacency = pmatrix#np.random.rand(ns,ns)<pmatrix
            adjacency_rand = np.random.rand(ns,ns)<pmatrix
            #degrees = np.sum(pmatrix>0.8,axis=0) + np.sum(pmatrix>0.999,axis=1)
            
            S[ss] = ns
            L[ss]= np.round(np.sum(adjacency))
            C[ss] = L[ss]/(ns**2)
            Lrand[ss]= np.sum(adjacency_rand)
            Crand[ss] = Lrand[ss]/(ns**2)

            
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(sampling_size,Lrand,'r')
        plt.plot(sampling_size,L,'b',linewidth=2.0)
        plt.ylabel('$L$',fontsize=20)        
        plt.xlabel('$n$',fontsize=20)        
        
        plt.subplot(1,2,2)
        plt.plot(sampling_size,Crand,'r')
        plt.plot(sampling_size,C,'b',linewidth=2.0)
        plt.ylabel('$C$',fontsize=20)        
        plt.xlabel('$n$',fontsize=20)        
        
        return L,S,C,sampling_size
        
           
         
         
    @staticmethod
    def PLOT_MODULAR_IG(gi):
        
        ig.plot(gi.community_leading_eigenvector())
        