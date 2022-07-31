import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from get_model_parameters import get_marriage_distances_kolton, test_find_distances_bio
nodes = [1, 2, 4, 5, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23]
parent_child_edges = [(22, 18), (22,19), (18, 14),
                      (19, 15), (14, 10),
                      (15, 11), (10, 4), (10, 5),
                      (11, 6), (11, 7), (4,1), (5,23),
                      (6, 23), (7, 2)]
marriage_edges = [(5,6), (1,2)]
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(parent_child_edges)


distances, num_inf_marriages, percent_inf_marraiges = get_marriage_distances_kolton(G, marriage_edges, plot=True)
distances
distances2, count2 = test_find_distances_bio(G, parent_child_edges, marriage_edges)
distances2
plt.hist(distances2, bins=[k for k in range(max(distances2) + 2)], range=(0, max(distances) + 2))
