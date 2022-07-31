import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from graph_attributes import get_graphs_and_names, separate_parts
from get_model_parameters_kolton import build_marriage_hist_kolton, get_marriage_distances_kolton, get_graphs_and_names, separate_parts
from time import time
import regex as re

names = ['anuta_1972',
         'tikopia_1930',
         'ammonni',
         'ancien_regime',
         'ragusa',
         'san_marino']

# shouldn't I make it so that this ONLY loads the graph in question?
# otherwise won't this be terribly wasteful memorywise?
graphs, graph_names = get_graphs_and_names(directed=True)
name_pattern = re.compile("(?<=-).*(?=-)")

count = 0  # number of graphs included in master histogram
master_distances = []
name = 'anuta_1972'
for name in graph_names:
    if name == "../Original_Sources/kinsources-warao-oregraph.paj":
        continue
    print(name)
    name = name_pattern.findall(name)[0]
    start = time()
    distances, num_inf_marriages, percent_inf_marraiges = build_marriage_hist_kolton(name)
    end = time()
    print('runtime: ', end-start)

    master_distances += distances
    count += 1
    try:
        with open('./Kolton_distances/'+name+'.txt', 'w') as outfile:
            outfile.write(str(distances))
            outfile.write('\n')
            outfile.write(str(num_inf_marriages))
            outfile.write('\n')
            outfile.write(str(percent_inf_marraiges))
            outfile.write('\n')
    except FileExistsError as e:
        print("./Kolton_distances/' +" + name + "+'.txt'file exists.  Skipping. ")

    g_num = graph_names.index('../Original_Sources/kinsources-'+name+'-oregraph.paj')
    vertex_names, marriage_edges, child_edges = separate_parts(graph_names[g_num], 'A')

    try:
        with open('./Kolton_distances_times/' + name + '.txt', 'w') as outfile:
            outfile.write('runtime: ' + str(end-start))
            outfile.write('\n')
            outfile.write('Number of nodes: ' +  str(len(vertex_names[1])))
            outfile.write('\n')
            outfile.write('Number of marriages: ' + str(len(marriage_edges)))
    except FileExistsError as e:
        print('./Kolton_distances_times/' + name + '.txt exists.  Skipping. ')
    print('\n')

print("count: ", count )
try:
    with open('./Kolton_distances/'+ "master_distances_104"+'.txt', 'w') as outfile:
        outfile.write(str(master_distances))
        outfile.write('\n')
except FileExistsError as e:
    print("./Kolton_distances/' +" + "master_distances_104"+ "+'.txt'file exists.  Skipping. ")
