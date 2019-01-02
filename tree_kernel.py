#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:50:04 2018

@author: raghebal-ghezi
"""
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from nltk import Tree
import numpy as np



def cos(a,b):
    # calc cosine similairty
    from numpy import dot
    from numpy.linalg import norm

    return dot(a, b)/(norm(a)*norm(b))



def parse(sentence, nltk_tree=False):
    '''
    Convert a sentence from a constitiuency tree format to list of binary tuples
    
    INPUT: "(S (NP I) (VP (V saw) (NP him)))"
    OUTPUT: [('S', 'NP'),('S', 'VP'), ('NP', "'I'"),('VP', 'V'),('VP', 'NP'),('V', "'saw'"),('NP', "'him'")]
    '''
    t = Tree.fromstring(sentence)
    
    if nltk_tree:
        t.draw()

    new_list = list()
    for i in t.productions(): 
       new_list.append(str(i).split(" "))
    for j in new_list:
        j.remove("->")
    nodes_list = list()   
    # flatten to binary tuple showing an edge between two nodes
    for i in new_list:
        if len(i) > 2:
            for j in i[1:]:
                nodes_list.append((i[0],j))
        else:
             nodes_list.append(tuple(i))
             
    return nodes_list 
#
parsed_tree = parse("(S (NP I) (VP (V saw) (NP him)))")

def visualize(parsed_tree):
    # takes list of binary tuples, visualize using matplotlib or PyDot
    tree2graph = nx.DiGraph()
    tree2graph.add_edges_from(parsed_tree)
    nx.draw(tree2graph,with_labels=True)
#    if pyDot_format:
#        p = nx.drawing.nx_pydot.to_pydot(tree2graph)
        

g = nx.DiGraph()
h = nx.DiGraph()

g.add_edges_from(parse("(S (NP I) (VP (V kill) (NP him)))"))
h.add_edges_from(parse("(S (NP I) (VP (V kill) (NP him)))"))
    

def kernel_tree(g,h):
    '''
    Implementation of Convolution Kernels (Collins and Duffy, 2002)
    Compute the cosine similarity over two trees
    INPUT: networkx DiGraph 
    OUTPUT: 0 is identical to 1 is different
    '''
            
    subtrees_G=[]
    subtrees_H=[]
    
    for i,node in enumerate(g.nodes()):
        subtrees_G.append(dfs_tree(g,node))
        
    for i,node in enumerate(h.nodes()):
        subtrees_H.append(dfs_tree(h,node))
    
    
    def label_check(d1,d2):
        return d1==d2

    all_subtrees=subtrees_G+subtrees_H
  
    v=[]
    w=[]
    for i,subtree in enumerate(all_subtrees):
        if subtree.nodes()!=[]:
            v.append(np.sum(np.array(list(map(lambda x: nx.is_isomorphic(subtree,x),subtrees_G)))))
            w.append(np.sum(np.array(list(map(lambda x: nx.is_isomorphic(subtree,x),subtrees_H)))))

#    print (v)
#    print (w)
    return cos(v,w)


print(kernel_tree(g,h)) 
