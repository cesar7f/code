# -*- coding: utf-8 -*-
"""
Created on Fri Feb 07 11:58:57 2014

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

def parallele_dictionary(kwargs):
    df_nodes,df_edges = DataManipulation.GET_DATASETS(*kwargs)
    return df_nodes,df_edges
    
def parallele_graphgml(kwargs):
    df_nodes,df_edges = DataManipulation.DF_TO_GRAPHML(*args,**kwargs)
    return df_nodes,df_edges

    
from ParallelProccessFactory import ParallelProccessFactory
        

class DataManipulation:
   
    _PRINT_STATUS = False   
   
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
    def GET_DATASETS(file_nodes='data/small_nodes.csv', file_edges='data/small_edges.csv',time_step=None,drop_sources=True):
        import pandas as pd
        df_nodes = pd.read_csv(file_nodes)
        print ' join '
        if(len(df_nodes.columns) == 1):
            print file_edges   
            
            df_nodes = pd.read_csv(file_nodes,sep='\t')
            df_edges = pd.read_csv(file_edges,sep='\t')
            
        else:
            df_edges = pd.read_csv(file_edges)
        
        if(time_step):
            df_nodes = df_nodes[df_nodes.TimeStep==time_step]
            df_edges = df_edges[df_edges.time_step==time_step]  
        
        df_edges.loc[df_edges.Prey_ID=='S1','Prey_ID'] = '-1'
        df_edges.loc[df_edges.Prey_ID=='S2','Prey_ID'] = '-2'
        df_edges[['Prey_ID']] = df_edges[['Prey_ID']].astype(int)        
        
        max_value =  10e12       
        
        df_nodes.loc[df_nodes['FunctionalGroup']=='S1','ID'] = -1
        df_nodes.loc[df_nodes['FunctionalGroup']=='S2','ID'] = -2
        df_nodes.loc[df_nodes['FunctionalGroup']=='S1','FunctionalGroup'] = 100
        df_nodes.loc[df_nodes['FunctionalGroup']=='S2','FunctionalGroup'] = 100

        df_nodes[['CohortAbundance']] = df_nodes[['CohortAbundance']].astype(float)
        df_nodes[['FunctionalGroup']] = df_nodes[['FunctionalGroup']].astype(int)
        
        #Assign values to the source nodes
        for column in df_nodes.dtypes.index:
            if(df_nodes.dtypes[column] == 'float'):
                df_nodes.loc[df_nodes['FunctionalGroup']==100,column] = max_value
        
        

        #df_nodes.set_index(['TimeStep','FunctionalGroup','ID'],inplace=True,drop=False)
        #df_edges.set_index(['time_step','Pred_ID'],inplace=True,drop=False)
        
        if(drop_sources):
            df_nodes = df_nodes[df_nodes['ID']>=0]
            
        df_edges = df_edges[df_edges['Prey_ID'].isin(df_nodes.ID.unique())]
        df_edges = df_edges[df_edges['Pred_ID'].isin(df_nodes.ID.unique())]
        
        


        
        
        return df_nodes,df_edges
        

    @staticmethod
    def ADD_LOGS_TO_DF(df):
        df = df.copy()
        map_types = {'int64':'integer','int':'integer','int32':'integer','str':'string',
                       'float':'double','float32':'double','float64':'double'}
        
        for attrib_name in df.dtypes.index:
                type_attrib = map_types[str(df.dtypes[attrib_name])]
                if (type_attrib == 'double'):
                    df['log10_'+attrib_name] = np.log10(df[attrib_name])
                    
        return df

    @staticmethod
    def DF_TO_GEPPHI_CSV(df_nodes,df_edges,file_nodes='nodes.csv',file_edges='edges.csv',add_logs = True, source = 'Prey_ID', target = 'Pred_ID', weigth_column = 'BiomassIngested'):
        #ATTRIBUTES DEFINITIONS
                       
        if(add_logs == True):
            df_nodes = DataManipulation.ADD_LOGS_TO_DF(df_nodes)
            df_edges = DataManipulation.ADD_LOGS_TO_DF(df_edges)            
                    
        df_edges['source'] = df_edges['Prey_ID']
        df_edges['target'] = df_edges['Pred_ID']
        df_edges['weight'] = np.log10(df_edges[weigth_column])
        
        df_nodes.to_csv(file_nodes)
        df_edges.to_csv(file_edges)
        
        
        

    @staticmethod       
    def DF_TO_GEXF(df_nodes,df_edges,directed=True,weigth_column='BiomassIngested',add_logs=True,file_output = None):
        
        edge_type = 'undirected'
        time_type = 'static'
        
        if directed: edge_type = 'directed'
        #if dynamic: time_type = 'dynamic'

        if(add_logs == True):
            df_nodes = DataManipulation.ADD_LOGS_TO_DF(df_nodes)
            df_edges = DataManipulation.ADD_LOGS_TO_DF(df_edges)  

        
        #ROOT DEFINITION
        xsi="http://www.w3.org/2001/XMLSchema-instance"
        viz="http://www.gexf.net/1.2draft/viz"
        root = etree.Element("gexf",attrib={'version':"1.2"},
                             nsmap={ None:"http://www.gexf.net/1.1draft",
                                    'viz':"http://www.gexf.net/1.2draft/viz",
                                    'xsi':"http://www.w3.org/2001/XMLSchema-instance"})
        root.set("{" + xsi + "}schemaLocation","http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd")
        
        
        #GRAPH DEFINITION
        graph = etree.SubElement(root, 'graph')
        graph.set('mode',time_type)
        graph.set('defaultedgetype',edge_type) #
        
        
        #ATTRIBUTES DEFINITIONS
        map_types = {'int64':'integer','int':'integer','int32':'integer','str':'string',
                       'float':'double','float32':'double','float64':'double'}
        map_casting = {'int64':int,'int':int,'int32':int,'str':str,
                       'float':float,'float32':float,'float64':float}
        #dypes extraction (improve a lot the speed)
        node_dtypes = df_nodes.dtypes
        edge_dtypes = df_edges.dtypes
        
        #For nodes             
        attributes = etree.SubElement(graph, 'attributes', attrib={'mode':time_type,'class':'node'})
        
       
        for attrib_name in node_dtypes.index:
            type_attrib = map_types[str(node_dtypes[attrib_name])]
            etree.SubElement(attributes, 'attribute', 
                                   attrib={'id':attrib_name, 'title':attrib_name,
                                   'type':type_attrib})

        #For edges
        attributes = etree.SubElement(graph, 'attributes', attrib={'mode':time_type,'class':'edge'})
        for attrib_name in df_edges.dtypes.index:
            type_attrib = map_types[str(edge_dtypes[attrib_name])]
            etree.SubElement(attributes, 'attribute', 
                                   attrib={'id':attrib_name, 'title':attrib_name,
                                   'type':type_attrib})     
                                   

        #NODE CREATION
        nodes = etree.SubElement(graph, 'nodes')
        colors = DataManipulation._FG_RGB_COLORS        
        ii = 0
        for idx in df_nodes.index:
            node_id = str(int(df_nodes.loc[idx,'ID']))
            node = etree.SubElement(nodes, 'node', attrib={'id':node_id})

            att_values = etree.SubElement(node, 'attvalues')
            for col_name in df_nodes.columns:
                val_attrib = df_nodes.loc[[idx],col_name].values[0]
                etree.SubElement(att_values, 'attvalue', attrib={'for':col_name,'value':str(val_attrib)})#
                    
            
            #Visual attributtes
            func_id = df_nodes.loc[[idx],'FunctionalGroup'].values[0]
            log_adult = 100*np.log10(df_nodes.loc[[idx],'AdultMass'].values[0])
            log_juvenile = 100*np.log10(df_nodes.loc[[idx],'JuvenileMass'].values[0])
            
            at_pos = {'x':str(log_adult),'y':str(log_juvenile),'z':str(0.0)}
            if(node_id == "-1"): at_pos = {'x':str(-100.0),'y':str(-100.0),'z':str(0.0)}
            if(node_id == "-2"): at_pos = {'x':str(-100.0),'y':str(-100.0),'z':str(0.0)}
            at_rgb = {'r':str(colors[func_id][0]),'g':str(colors[func_id][1]),'b':str(colors[func_id][2])}
            etree.SubElement(node, "{" + viz + "}color", attrib=at_rgb)
            etree.SubElement(node, "{" + viz + "}position", attrib=at_pos)#
            ii += 1
            if ii % 100 == 0: print 'Node %i analyzed' % ii
            

        #EDGE ATTRIBUTE DEFINITION
        edges = etree.SubElement(graph, 'edges')
        ii = 0
        for index,row in df_edges.iterrows():
            edge = etree.SubElement(edges, 'edge', 
                                    attrib= {'source':str(int(row['Prey_ID'])),'target':str(int(row['Pred_ID'])),
                                             'weight':str(np.log10(row[weigth_column]))})

            att_values = etree.SubElement(edge, 'attvalues')
            for col_name in df_edges.columns:
                val_attrib = map_casting[str(edge_dtypes[col_name])](row[col_name])
                etree.SubElement(att_values, 'attvalue', attrib={'for':col_name,'value':str(val_attrib)})#
            
            ii += 1
            if (ii % 1000 == 0) and DataManipulation._PRINT_STATUS: print 'Edge %i analyzed' % ii

        if file_output:
            DataManipulation.PRINT_STR_TO_FILE(etree.tostring(root, pretty_print=True),file_output)
        else:
            etree.tostring(root, pretty_print=True)
            
        return root
            


    @staticmethod       
    def DF_TO_GRAPHML(df_nodes,df_edges,directed=True,weigth_column='BiomassIngested',add_logs=True,file_output = None):
        
        edge_type = 'undirected'
        time_type = 'static'
        
        if directed: edge_type = 'directed'
        #if dynamic: time_type = 'dynamic'

        if(add_logs == True):
            df_nodes = DataManipulation.ADD_LOGS_TO_DF(df_nodes)
            df_edges = DataManipulation.ADD_LOGS_TO_DF(df_edges)  
        
        
        #ROOT DEFINITION        
        graph_gml = etree.Element('graphml',attrib={'xmlns':'http://graphml.graphdrawing.org/xmlns'})
        
        
        #ATTRIBUTES DEFINITIONS
        map_types = {'int64':'long','int':'long','int32':'long','str':'string',
                       'float':'double','float32':'double','float64':'double'}
        map_casting = {'int64':int,'int':int,'int32':int,'str':str,
                       'float':float,'float32':float,'float64':float}
        #dypes extraction (improve a lot the speed)
        node_dtypes = df_nodes.dtypes
        edge_dtypes = df_edges.dtypes
        
        extra_node_attributes = {'x':'float','y':'float','z':'float','r':'int','g':'int','b':'int'}
        extra_edge_attributes = {'weight':'double'}
        
        #For nodes
        for attrib_name in node_dtypes.index:
            type_attrib = map_types[str(node_dtypes[attrib_name])]
            etree.SubElement(graph_gml, 'key', 
                                   attrib={'id':attrib_name, 'attr.name':attrib_name,
                                   'attr.type':type_attrib, 'for':'node'})
        for attrib_name, type_attrib in extra_node_attributes.items():
            etree.SubElement(graph_gml, 'key', 
                                   attrib={'id':attrib_name, 'attr.name':attrib_name,
                                   'attr.type':type_attrib, 'for':'node'})
                                   
        #For edges
        for attrib_name in df_edges.dtypes.index:
            type_attrib = map_types[str(edge_dtypes[attrib_name])]
            etree.SubElement(graph_gml, 'key', 
                                   attrib={'id':attrib_name, 'attr.name':attrib_name,
                                   'attr.type':type_attrib, 'for':'edge'})
        for attrib_name, type_attrib in extra_edge_attributes.items():
            etree.SubElement(graph_gml, 'key', 
                                   attrib={'id':attrib_name, 'attr.name':attrib_name,
                                   'attr.type':type_attrib, 'for':'edge'})                                   
                                           
        graph = etree.SubElement(graph_gml, 'graph',attrib={'edgedefault':'directed'})
        
        #NODE CREATION
        colors = DataManipulation._FG_RGB_COLORS        
        ii = 0
        for idx in df_nodes.index:
            node_id = str(int(df_nodes.loc[idx,'ID']))
            node = etree.SubElement(graph, 'node', attrib={'id':node_id})

            for col_name in df_nodes.columns:
                val_attrib = df_nodes.loc[[idx],col_name].values[0]
                node_attrib = etree.SubElement(node, 'data', attrib={'key':col_name})#
                node_attrib.text = str(val_attrib)
                    
        
            #Visual attributtes
            func_id = df_nodes.loc[[idx],'FunctionalGroup'].values[0]
            log_adult = 100*np.log10(df_nodes.loc[[idx],'AdultMass'].values[0])
            log_juvenile = 100*np.log10(df_nodes.loc[[idx],'JuvenileMass'].values[0])
            
            at_pos = {'x':str(log_adult),'y':str(log_juvenile),'z':str(0.0)}
            if(node_id == "-1"): at_pos = {'x':str(-100.0),'y':str(-100.0),'z':str(0.0)}
            if(node_id == "-2"): at_pos = {'x':str(-100.0),'y':str(-100.0),'z':str(0.0)}
            at_rgb = {'r':str(colors[func_id][0]),'g':str(colors[func_id][1]),'b':str(colors[func_id][2])}
            
            node_attrib = etree.SubElement(node, 'data', attrib={'key':'r'})#
            node_attrib.text = at_rgb['r']
            node_attrib = etree.SubElement(node, 'data', attrib={'key':'g'})#
            node_attrib.text = at_rgb['g']
            node_attrib = etree.SubElement(node, 'data', attrib={'key':'b'})#
            node_attrib.text = at_rgb['b']
            
            node_attrib = etree.SubElement(node, 'data', attrib={'key':'x'})#
            node_attrib.text = at_pos['x']
            node_attrib = etree.SubElement(node, 'data', attrib={'key':'y'})#
            node_attrib.text = at_pos['y']
            
            ii += 1
            if ii % 100 == 0 and DataManipulation._PRINT_STATUS: print 'Node %i analyzed' % ii
            
        

        #EDGE ATTRIBUTE DEFINITION        
        ii = 0
        for index,row in df_edges.iterrows():
            edge = etree.SubElement(graph, 'edge',
                                    attrib= {'source':str(int(row['Prey_ID'])),'target':str(int(row['Pred_ID']))})
            
            edge_attrib = etree.SubElement(edge, 'data', attrib= {'key':'weight'})
            edge_attrib.text = str(np.log10(row[weigth_column]))               
            for col_name in df_edges.columns:
                val_attrib = map_casting[str(edge_dtypes[col_name])](row[col_name])
                edge_attrib = etree.SubElement(edge, 'data', attrib= {'key':col_name})
                edge_attrib.text = str(val_attrib)

            
            ii += 1
            if ii % 1000 == 0 and DataManipulation._PRINT_STATUS: print 'Edge %i analyzed' % ii

        if file_output:
            DataManipulation.PRINT_STR_TO_FILE(etree.tostring(graph_gml, pretty_print=True,xml_declaration=False),file_output)
        else:
            print etree.tostring(graph_gml, pretty_print=True, xml_declaration=False)
            
        return graph_gml
       
    @staticmethod        
    def PRINT_STR_TO_FILE(string_to_print,file_name):
        
        text_file = open(file_name, "w")
        text_file.write(string_to_print)
        text_file.close()  
        
    @staticmethod    
    def CREATE_ALL_GRAPHML_FROM_FILES(nodes_dir='data/nodes/',edges_dir='data/edges/',time_step=1223,output_dir='data/graphml/'):
        
        nodes,edges = DataManipulation.GET_BIG_DICTIONARY(nodes_dir='data/nodes/',edges_dir='data/edges/',time_step=1223)
        
        for key in nodes.keys():
            file_name = output_dir+('graphml_%f.xml' % key)
            print 'Creating file %s' % file_name
            DataManipulation.DF_TO_GRAPHML(nodes[key], edges[key], file_output = file_name)
            
            

    @staticmethod    
    def GET_BIG_DICTIONARY(nodes_dir='data/nodes/',edges_dir='data/edges/',time_step=1223,run_in_parallel = False):
        '''Be sure that you have the same number of files inside nodes_dir and edges_dir and that
        sorting will give you the same correspondance order in both directories. This is because
        this function will map each node file with each edge file'''

        node_files = [nodes_dir+f for f in listdir(nodes_dir) if isfile(join(nodes_dir,f)) ]
        edge_files = [edges_dir+f for f in listdir(edges_dir) if isfile(join(edges_dir,f)) ]

        #You need both node and edge data sets in the same ammount        
        assert(len(node_files) == len(edge_files))           
        
        node_files.sort()
        edge_files.sort()
        
        nodes = dict()
        edges = dict()
        
        keys = [float(nf.split('_')[1]) for nf in node_files]
                    
        try:
            if not run_in_parallel:
                raise Exception("I know python!") ## this parallel thing does not work well yet, therefore avoid using it by the moment
            paral = ParallelProccessFactory(parallele_dictionary)
            paral.pmap(zip(node_files,edge_files,len(node_files)*[time_step]))
            for i in range(len(node_files)):
                nodes[keys[i]] = paral.map_results[i][0]
                edges[keys[i]] = paral.map_results[i][1]
            
        except:
            print 'COULD NOT PARALLALIZE, DO SEQUENTIALLY INSTEAD...'
            for f_node,f_edges in zip(node_files, edge_files):
                node_key = float(f_node.split('_')[1])
                edge_key = float(f_edges.split('_')[1])
                nodes[node_key],edges[edge_key] = DataManipulation.GET_DATASETS(file_nodes=f_node, file_edges=f_edges,time_step=time_step,drop_sources=True)
        
        return nodes,edges
        
        