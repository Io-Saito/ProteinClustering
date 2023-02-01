"""
title: ProteinClustering
description: Leidenalg Clustering of the Enhanced Protein Covariation Network
author: Io V Saito

This script does the work of clustering a network provided as an adjacency
matrix using the Leiden algorithm and the Python Leidenalg library. 

Leidenalg is the work of Vincent Traag:
http://dx.doi.org/10.1038/s41598-019-41695-z
"""


from sys import stderr

import numpy as np
from importlib import import_module
from leidenalg import Optimiser, find_partition

from igraph import Graph
import pandas as pd
from tqdm import tqdm


class LeidenAlg():
    def __init__(self,parameters:dict,recursive=False,recursive_parameters=None,max_size=100):
        self.parameters = parameters
        self.g=None
        self.adjm=None
        self.max_size=max_size
        self.recursive_params=recursive_parameters
        self.profile=[]
        self.recursive=recursive
       
    
    def Run(self,adjm)->pd.DataFrame:
        print(f"Performing Leidenalg clustering utilizing the {self.parameters['method']}", file=stderr)
        self.adjm2Graph(adjm)
        if self.recursive:
            print(
                f"Recursively splitting modules larger than {self.max_size} nodes with {self.recursive_params['method']}", file=stderr)
        if self.parameters["multi_resolution"]:
            self.res_range = np.linspace(self.parameters["resolution_parameter"])
            print(
                f"Analyzing graph at {len(self.res_range)} resolutions.", file=stderr)
            self.MultiRes()
        else:
            self.SingleRes()
        self.dataframe = pd.DataFrame(columns=self.profile[0].graph.vs['name'])
        for i in range(len(self.profile)):
            self.dataframe.loc[i] = self.profile[i].membership
        return self.dataframe

    def postprocess(self,min_value=5):
        df_partition = self.dataframe.T[1:]
        df_partition.columns = ["A", "B"]
        x = df_partition["A"].value_counts() > min_value
        self.df_postprocessed = df_partition.where(
        df_partition["A"].isin(x[x].index.to_list()), 0)
        return self.df_postprocessed
        
    def adjm2Graph(self,adjm):
        if self.parameters["signed"] == False:
            self.adjm = abs(adjm)
        else:
            self.adjm = adjm
        edges = self.adjm.stack().reset_index()
        edges.columns = ['nodeA', 'nodeB', 'weight']
        edges = edges[edges.weight != 0]
        edge_tuples = list(zip(edges.nodeA, edges.nodeB, edges.weight))
        if self.parameters["weights"] == True:
            self.g = Graph.TupleList(edge_tuples, weights=True)
        else:
            self.g = Graph.TupleList(edge_tuples, weights=False)
        print("Input graph: {}".format(self.g.summary()), file=stderr)

    def Optim(self,graph,params):
        partition_type = getattr(import_module(
            'leidenalg'), params["partition_type"])
        partition = find_partition(graph=graph,partition_type=partition_type,n_iterations=params["n_iterations"])
        optimiser = Optimiser()
        diff = optimiser.optimise_partition(
            partition, params["n_iterations"])
        print(f"Partition summary:{partition.summary()}",file=stderr)
        return partition

    def GetSubgraph(self,partition,new_params):
        subgraphs = partition.subgraphs()
        too_big = [subg.vcount() > self.max_size for subg in subgraphs]
        while any(too_big):
            # Perform clustering for any subgraphs that are too big.
            idx = [i for i, too_big in enumerate(too_big) if too_big]
            new_graph = subgraphs.pop(idx[0])
            # mask negative edges
            edge_weights = new_graph.es['weight']
            new_weights = [e if e > 0 else 0 for e in edge_weights]
            new_graph.es['weight'] = new_weights
            new_params['graph'] = new_graph
            new_params['n_iteration']=-1
            part=self.Optim(new_graph,new_params)
            subgraphs.extend(part.subgraphs())
            too_big = [subg.vcount() > self.max_size for subg in subgraphs]
        return subgraphs
    
    
    def MultiRes(self):
        for resolution in tqdm(self.res_range):
            self.parameters['resolution_parameter'] = resolution
            self.SingleRes()


    def SingleRes(self):
        partition = self.Optim(self.g,self.parameters)
        self.profile.append(partition)
        if self.recursive:
        # update clustering params
            new_params=self.recursive_params
            subgraphs=self.GetSubgraph(partition,new_params)
            nodes = [subg.vs['name'] for subg in subgraphs]
            parts = [dict(zip(n, [i]*len(n))) for i, n in enumerate(nodes)]
            new_part = {k: v for d in parts for k, v in d.items()}
        # Set membership of initial graph.
            membership = [new_part.get(node)
                      for node in partition.graph.vs['name']]
            partition.set_membership(membership)
            self.profile.append(partition)
