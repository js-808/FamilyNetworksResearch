import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph_attributes import *
from scipy.stats import linregress
import pandas as pd
import pyarrow.feather as feather
import ast
import random


def find_distances_bio(g_num, graph_names):
    """Finds length of shortest 'biological' path between each marriage node in a given graph (i.e., path goes through parents)

    Parameters: g_num (int): graph number in list of graphs

    Returns: distances (list): list of union distances
             count (int): number of infinite union distances
    """
    print(graph_names[g_num])
    og_graph = graph_with_attributes(graph_names[g_num])

    # get all parts of graph
    stuff = separate_parts(graph_names[g_num],'A')

    # list of marriage edge tuples for the given graph
    marriages = stuff[1]

    # list of parent child edge tuples for the given graph
    children = stuff[2]

    distances = []
    count = 0

    # delete all marriage edges first
    for i in range(len(marriages)):

        # get parent nodes
        p1, p2 = marriages[i]

        # delete marriage edge
        og_graph.remove_edge(p1,p2)

    # go through all marriage pairs
    for i, m in enumerate(marriages):
        # copy graph to change temporarily
        g = og_graph.copy()

        # get parent nodes
        p1, p2 = m

        # find children of each parent node & delete children nodes from parent nodes
        for child in children:
            if child[0] == p1:
                g.remove_edge(p1,child[1])
            elif child[0] == p2:
                g.remove_edge(p2,child[1])

        try:
            # find shortest path between parent nodes
            path = nx.shortest_path(g, source=p1, target=p2)

            # record length of shortest path
            distances.append(len(path)-1)

        except nx.NetworkXNoPath:
            # count number of marriage nodes that don't have path between them
            count+=1

    return distances, count


def find_children(g_num, graph_names):
    og_graph = graph_with_attributes(graph_names[g_num])

    # get all parts of graph
    stuff = separate_parts(graph_names[g_num],'A')

    # list of marriage edge tuples for the given graph
    marriages = stuff[1]

    # list of parent child edge tuples for the given graph
    children = stuff[2]

    count = []

    # go through all marriage pairs
    for i, m in enumerate(marriages):

        # get parent nodes
        p1, p2 = m

        p1_list = set()
        p2_list = set()

        # find children of each parent node & count children node
        for child in children:
            if child[0] == p1:
                p1_list.add(child[1])
            elif child[0] == p2:
                p2_list.add(child[1])

        # union of p lists
        #p1_list |= p2_list

        #intersection of p lists
        c_list = p1_list.intersection(p2_list)

        count.append(len(c_list))

    return count


def get_some_parameters_old(g_num, name):
    """Get necessary parameters for model
    Parameters:
                g_num (int): genealogical network number (i.e., the dataset you want to use for the model)
                name (str): name used to differentiate files (should be descriptive of chosen genealogical network)

    Returns:
                m_e (int): number of marriage edges in genealogical network
                inf (int): number of infinite distance marriages
                P (float): probability of marriage
                NCP (float): probability of non-connected marriage
    """

    # load all original sources
    graphs, graph_names = get_graphs_and_names()   # graphs is a list of all Kinsource graphs
                                                   # graph_names is a list of Kinsource graph file names

    # genealogical network
    G = graphs[g_num]

    # get total number of nodes in network
    total = G.number_of_nodes()

    # Gives the number of males, females, unknown, marriage edges, parent-child edges in network
    attribs = count_attributes(G)

    # get number of marriage edges in network
    m_e = attribs[3]

    # get probability of marriage
    P = m_e*2/total

    # get all parts of graph
    stuff = separate_parts(graph_names[g_num],'A')

    # list of marriage edge tuples for the given network
    marriages = stuff[1]

    # list of parent-child edge tuples for the given network
    children = stuff[2]

    # get set of nodes that are married
    all_married = set()
    for pair in marriages:
        all_married.add(pair[0])
        all_married.add(pair[1])
    all_married  # set of all nodes that are married

    # get set of nodes that are children
    all_children = set()
    for pc in children:
        child = pc[1]
        all_children.add(child)
    all_children  # set of all nodes that are children

    # get nodes that are married but are not children
    not_children = all_married - all_children

    # get probability of non-connected marriage
    NCP = len(not_children)/len(all_married)

    # get list of finite distances and number of infinite distances
    finite_dis, inf_dis = find_distances_bio(g_num, graph_names)

    # Save list of finite distances as a txt file
    text_file = open("{}_distances.txt".format(name), "w")
    n = text_file.write(str(finite_dis))
    text_file.close()


    # get number of children from each union
    num_children = find_children(g_num, graph_names)

    # Save list of number of children as a txt file
    text_file = open("{}_children.txt".format(name), "w")
    n = text_file.write(str(num_children))
    text_file.close()

    return m_e, P, NCP, inf_dis

# name = 'tikopia_1930'
def get_some_parameters(name):
    """Get necessary parameters for model
    Parameters:
                name (str): name of chosen network from the Kinsource data

    Returns:
                m_e (int): number of marriage edges in genealogical network
                P (float): probability of marriage
                NCP (float): probability of non-connected marriage
    """

    # load all original sources
    graphs, graph_names = get_graphs_and_names()   # graphs is a list of all Kinsource graphs
                                                   # graph_names is a list of Kinsource graph file names
    # get number of chosen graph
    g_num = graph_names.index('./Original_Sources/kinsources-'+name+'-oregraph.paj')

    # genealogical network
    G = graphs[g_num]

    # get total number of nodes in network
    total = G.number_of_nodes()

    # Gives the number of males, females, unknown, marriage edges, parent-child edges in network
    attribs = count_attributes(G)

    # get number of marriage edges in network
    m_e = attribs[3]

    # get probability of marriage
    P = m_e*2/total

    # get all parts of graph
    stuff = separate_parts(graph_names[g_num],'A')

    # list of marriage edge tuples for the given network
    marriages = stuff[1]

    # list of parent-child edge tuples for the given network
    children = stuff[2]

    # get set of nodes that are married
    all_married = set()
    for pair in marriages:
        all_married.add(pair[0])
        all_married.add(pair[1])
    all_married  # set of all nodes that are married

    # get set of nodes that are children
    all_children = set()
    for pc in children:
        child = pc[1]
        all_children.add(child)
    all_children  # set of all nodes that are children

    # get nodes that are married but are not children
    not_children = all_married - all_children

    # get probability of non-connected marriage
    NCP = len(not_children)/len(all_married)

    return m_e, P, NCP


def get_marriage_distances_kolton(G, marriage_edges, name='', plot=True):
    """
    finds the initial marriage distance distrubtion (IE distance to nearest
    common ancestor) for a given genealogical network.  Assumes that siblings
    are distance 2 apart.

    PARAMETERS:
    G: (networkx digraph):  should comprise ALL nodes in the given community and
                            all parent-child edges (directed FROM parent TO
                            child) but should NOT contain ANY marriage edges (IE
                            those between spouses)
    marraige_edges (list of tuples):  list of marriages in the network (man,
                            wife)

    RETURNS:
    distances: (list) of len(marriage_edges); each entry is the distance from
                            one spouse, through the nearest common ancestor, to
                            the other spouse in the corresponding entry of
                            marraige_edges.  Infinite distance marriages (when
                            no common ancestor exists within the network) are
                            assigned a distance of 999.
    num_inf_marriages: (int) number of infinite distance (distance of 999)
                            marriages in the genealogical network.
    percent_inf_marraiges: (float) proportion of number of infinite distance
                            (distance of 999) marriages to the total number of
                            marriages in the genealogical network.
    """
    distances = []

    for couple in marriage_edges:
        # get parents
        paternal_gen = {couple[0]}
        maternal_gen = {couple[1]}
        paternal_tree = paternal_gen.copy()
        maternal_tree = maternal_gen.copy()
        intersection = paternal_tree.intersection(maternal_tree)
        dist = 0
        paternal_distances = {couple[0]:dist}
        maternal_distances = {couple[1]:dist}
        while len(intersection) == 0:

            dist += 1
            paternal_gen = set([parent[0] for ancestor in paternal_gen for parent in G.in_edges(ancestor)])
            maternal_gen = set([parent[0] for ancestor in maternal_gen for parent in G.in_edges(ancestor)])
            paternal_distances = paternal_distances | {ancestor: dist for ancestor in paternal_gen}
            maternal_distances = maternal_distances | {ancestor: dist for ancestor in maternal_gen}
            paternal_tree = paternal_tree.union(paternal_gen)
            maternal_tree = maternal_tree.union(maternal_gen)
            intersection = paternal_tree.intersection(maternal_tree)
            if len(paternal_gen) == 0 and len(maternal_gen) == 0 and len(intersection) == 0:
                # IE if both trees are exhausted, no new parents
                dist = 999  # IE infinite distance marraige
                break
        if dist == 999:
            distances.append(dist)
        else:
            min_dist = min([paternal_distances[common_ancestor] + maternal_distances[common_ancestor] for common_ancestor in intersection])
            distances.append(min_dist)

    distances_array = np.array(distances)
    num_inf_marriages = sum(distances_array == 999)
    percent_inf_marraiges = num_inf_marriages/len(marriage_edges)
    if plot and len(distances_array[distances_array != 999]) != 0:
        max_bin = np.max(distances_array[distances_array != 999])
        plt.hist(distances, bins=[k for k in range(max_bin + 2)], range=(0, max_bin+2))
        title = name + '\n'
        plt.title(title + "You have {} inf-distance marriages ({}%)".format(num_inf_marriages, round(percent_inf_marraiges, 3)*100))
        plt.show()
    if len(distances_array[distances_array != 999]) == 0:
        print("OJO! All distances were infinite!")
    return distances, num_inf_marriages, percent_inf_marraiges

# distances, num_inf_marriages, percent_inf_marraiges = get_marriage_distances_kolton(G, marriage_edges, plot=True)
# name = 'shoshone_1880_nd11'
# name ='anuta_1972'
# name = 'tikopia_1930'
# load all original sources
graphs, graph_names = get_graphs_and_names(directed=True)   # graphs is a list of all Kinsource graphs
                                               # graph_names is a list of Kinsource graph file names
def build_marriage_hist_kolton(name):
    # # load all original sources
    # graphs, graph_names = get_graphs_and_names(directed=True)   # graphs is a list of all Kinsource graphs
    #                                                # graph_names is a list of Kinsource graph file names
    # get number of chosen graph
    g_num = graph_names.index('./Original_Sources/kinsources-'+name+'-oregraph.paj')

    # genealogical network
    G = graphs[g_num]

    # get all parts of graph
    vertex_names, marriage_edges, child_edges = separate_parts(graph_names[g_num],'A')
    distances, num_inf_marriages, percent_inf_marraiges = get_marriage_distances_kolton(G, marriage_edges, name=name, plot=True)
    return distances, num_inf_marriages, percent_inf_marraiges


# len(marriage_edges)
# len(child_edges)
# # distances, num_inf_marriages, percent_inf_marraiges = build_marriage_hist_kolton('tikopia_1930')
# distances, num_inf_marriages, percent_inf_marraiges = build_marriage_hist_kolton('ojibwa_1930_nd07')
# distances, num_inf_marriages, percent_inf_marraiges = build_marriage_hist_kolton('shoshone_1880_nd11')
#
# with open('./UnionDistances/{}_distances.txt'.format(name)) as f:
#     nx_dis = f.readline()
# # network's marriage distances w/o infinite distances distribution
# marriage_dist = ast.literal_eval(nx_dis)
# marriage_dist = np.array(marriage_dist)
# max_bin = np.max(marriage_dist[marriage_dist != 100])
# num_inf_marriages = sum(marriage_dist == 100)
# percent_inf_marraiges = num_inf_marriages/len(marriage_edges)
# plt.hist(marriage_dist, bins=[k for k in range(max_bin)], range=(0, max_bin))
# plt.show()
# distances, num_inf_marriages, percent_inf_marraiges = build_marriage_hist_kolton('tikopia_1930')
#
#
# # synthetic tree to test distance functions
# nodes = [k for k in range(1,10)]
# marriages = [(9,2), (3,8), (4,6), (5,7)]
# parent_child_edges = [(1,2), (1,3), (2, 4), (2,5), (9, 4), (9,5), (3, 6), (3,7), (8,6), (8,7)]  # (parent, child)
# G = nx.DiGraph()
# G.add_nodes_from(nodes)
# G.add_edges_from(parent_child_edges)
# G.edges
# nx.draw_networkx(G, pos=nx.spring_layout(G))
# get_marriage_distances_kolton(G, marriages)


"""
This code below is adapted from find_distances_bio() to test on the synthetic tree above.
NOTE:  this code agrees with my code...
"""
def test_find_distances_bio(G, parent_child_edges, marriages):
    distances = []
    count = 0

    for i, m in enumerate(marriages):
        # copy graph to change temporarily
        g = G.to_undirected().copy()

        # get parent nodes
        p1, p2 = m

        # find children of each parent node & delete children nodes from parent nodes
        for child in parent_child_edges:
            if child[0] == p1:
                g.remove_edge(p1,child[1])
            elif child[0] == p2:
                g.remove_edge(p2,child[1])

        try:
            # find shortest path between parent nodes
            path = nx.shortest_path(g, source=p1, target=p2)

            # record length of shortest path
            distances.append(len(path)-1)

        except nx.NetworkXNoPath:
            # count number of marriage nodes that don't have path between them
            count+=1
    return distances, count

# nodes = [k for k in range(1,8)]
# marriages = [(2,3), (4,6), (5,7)]
# parent_child_edges = [(2,1),(3,1), (4,2), (6,2), (5,3), (7,3)]  # (parent, child)
# G = nx.DiGraph()
# G.add_nodes_from(nodes)
# G.add_edges_from(parent_child_edges)
# G.edges


# plt.figure(1, figsize=(25, 25))
# pos = nx.spring_layout(G)
# nx.draw_networkx(G, pos=pos)
# nx.draw_networkx(G.subgraph(couple), pos=pos, node_color='red')
# plt.show()


# get_marriage_distances_kolton(G, marriages)
# test_find_distances_bio(G, parent_child_edges)
#
#
# build_marriage_hist_kolton(name)
#
# tikopia = graphs[g_num]
# test_find_distances_bio(tikopia, children)
# find_distances_bio(g_num, graph_names)
