import networkx as nx
import random
import functools
import operator
import numpy as np
import ast
from scipy import stats

from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity as KDE
from scipy import interpolate

import itertools
import pickle
from time import time
from functools import wraps
import os
import regex as re


def makeOutputDirectory(out_directory, name):
    """
    Make an output directory to keep things cleaner

    Returns a full output path to the new directory
    """
    ver = 1
    output_dir = os.path.join(out_directory, name + '_')
    while os.path.exists(output_dir+str(ver)):
        ver += 1
    output_dir += str(ver)
    os.makedirs(output_dir)
    return output_dir


def get_graph_path(name, path='../Original_Sources/'):
    """
    PARAMETERS:
        name: (str) the name of the kinsources data set (see below for format)
    RETURNS:
        path: (str) path to directory prepended to the full name of specified
              kinsources file
    """
    return path + 'kinsources-'+name+'-oregraph.paj'


def get_num_people(name):
    """
    gets the number of people (total number of verticies) in the .paj file
    This function assumes that the number of nodes is correctly reported in the
    3rd line (index 2) of the .paj file. For example, contents should begin in
    the following format (defined in the funtion below):
        ['*Network Ore graph Tikopia.puc\n',
         '\n',
         '*vertices 294\n',
         "1 'X (1)' ellipse\n",
         ...]

    PARAMETERS:
        name: (str) the name of the kinsources data set
    RETURNS:
        num_people: (int) total number of people in the graph
    """
    path_to_graph = get_graph_path(name)
    # open and read graph file
    with open(path_to_graph, 'r') as file:
        contents = file.readlines()

    num_people = contents[2]
    num_people_pattern = re.compile("[0-9]+")
    num_people = int(num_people_pattern.findall(num_people)[0])

    return num_people


def get_graph_stats(name, distance_path='./Kolton_distances/', child_number_path='../ChildrenNumber/'):
    """
    Gets the statistics of a specified kinsources dataset
    PARAMETERS:
        name: (str) the name of the kinsources data set
        distance_path: (str) the filepath to the directory containing the saved
            text files containing the distance to marriage distributions (the output
            of timing_kolton_distance_algorithm.py)
        child_number_path: (str) the filepath to the directory containing the
            saved text files containing the children per couple distributions
    RETURNS:
        marriage_dists: (list of int) one entry per marriage indicating how many
            generations between spouses (reported in the number of parent-child
            edges crossed so that distance between siblings is 2) in the
            specified dataset.  If no common ancestor (IE an infinite distance)
            then the corresponding entry is -1
        num_marriages: (int) total number of marriage edges in the specified
            dataset
        prob_inf_marriage: (float) number of infinite marraiges divided by total
            number of marriages in the specified dataset
        prob_finite_marriage: (float) number of non-infinite marriages divided
            by total number of marriages in the specified dataset
        children_dist: (list of int) one entry per pair of parents, indicating
            how many child edges each parent in the couple share
        num_people: (int) total number of nodes in the graph named.
    """
    with open(distance_path + '{}.txt'.format(name)) as infile:
        marriage_dists, num_inf_marriages, fraction_inf_marriage = [ast.literal_eval(k.strip()) for k in infile.readlines()]

    # number of children data of chosen network
    with open(child_number_path + '{}_children.txt'.format(name)) as f:
        nx_child = f.readline()
    children_dist = ast.literal_eval(nx_child)

    num_marriages = len(marriage_dists)
    num_people = get_num_people(name)
    prob_marriage = num_marriages * 2 / num_people  # *2 since 2 spouses per marriage
    prob_inf_marriage = prob_marriage * fraction_inf_marriage
    prob_finite_marriage = prob_marriage - prob_inf_marriage

    return marriage_dists, num_marriages, prob_inf_marriage, prob_finite_marriage, children_dist, num_people


def get_probabilities(data, bandwidth=1, is_child=False):
    """
    given a list (either of distances to marriage for a marriage distribution or
    numbers of children for a children distribution), get_probabilities() produces
    a dictionary of probabilities. NOTE: the resulting "probabilities" should
    not be expected to sum to 1.  If a true probability distribution is desired
    then you should normalize the resulting distribution.  The resulting
    dictionary has entries beyond the data supplied (for example if a supplied
    marriage distribution has a maximum distance of 14 the resulting dictionary
    has entries for distances greater than 14 to allow us to use the
    datastructure without key errors should a larger number be drawn; we add
    1000 entries beyond the maximum.  If ever more than 1000 generations are to
    be run in the model, then this function should be modified).
    PARAMETERS:
        data (list): data taken from an actual family network
        bandwidth (int):  used as an argument in to_KDE(), the std deviation of
            each kernel in the sum (see documentation)
    RETURNS:
        probs (dictionary): keys are the entries of data and successive values,
            too (we lengthen the right tail of the distribution).
    """
    # # ??? should this data list NOT contain the infinite entries?
    # #     currently includes infinite entries (under key 999)
    # #     would it be better to have infinity represented as a distance of 0?
    # data = np.array(data)
    # data = data[data > -1]  # only use the non-infinite distances
    # kde = to_KDE(data, bandwidth)
    # domain = np.arange(min(data)-1, max(data)+1000, 1)  # ??? shouldn't I go from 0 to inf or from 2 to inf all the time?
    # domain = domain[:, np.newaxis]
    # logs = kde.score_samples(domain)
    # y = np.exp(logs)
    #
    # # fit spline (IE fit equation to density curve to then be able to integrate)
    # spl = interpolate.InterpolatedUnivariateSpline(domain, y)
    #
    # # create a dictionary of probabilities by integrating the density curve
    # probs = {i:spl.integral(i-0.5, i+0.5) for i in range(min(data), max(data)+1000)}
    # return probs
    data = np.array(data)
    data = data[data > -1]  # only use the non-infinite distances
    if is_child:  # IE is child
        domain = np.arange(0, max(data)+1000, 1)
    else:
        # ??? I feel like marriage distances should always have all distances
        #     possible, even the gross ones (IE we need to count from 2 not the
        #     minimum distance seen in the dataset)
        # domain = np.arange(2, max(data)+1000, 1)
        domain = np.arange(min(data)-1, max(data)+1000, 1)  # ??? shouldn't I go from 0 to inf or from 2 to inf all the time?

    kde = gaussian_kde(data, bw_method=bandwidth)
    # probs = {x:kde2.integrate_box_1d(-np.inf, x) for x in domain}  # CDF, discretized
    probs = {x:kde.integrate_box_1d(x-0.5, x+0.5) for x in domain}
    return probs


# TODO: it would be better to make this accept marriage_probs as an argument rather than
#       using marraige_probs as a global argument, right?
# TODO: need to make it so that the model runs, saves, and exits with an error message if people should ever be an empty list
# people=generation_of_people
# prob_marry_immigrant = prob_inf_marriage
# prob_marry = prob_finite_marriage


def kolton_add_marriage_edges(people, finite_marriage_probs, prob_marry_immigrant, prob_marry, D, indices, tol=1e-7):
    """
    Forms both infinite and finite distance marriages in the current generation
    PARAMETERS:
        people:  (list) of the current generation (IE those people elligible for
                marriage)
        finite_marriage_probs: (dictionary) keys are marriage distances, values
            are probabilities.  Note that this dictionary should only include
            entries for NON-inifite marriage distances, should have
            non-negaitive values which sum to 1, and should have a long right
            tail (IE lots of entries which map high (beyond what occurs in the
            example dataset) distances to zero(ish) probabilties)
        prob_marry_immigrant: (float) the probablility that a given node will marry
                a immigrant (herein a person from outside the genealogical network,
                without comon ancestor and therefore at distance infinity from the
                nodes in the list 'people') (formerly 'ncp')
        prob_marry: (float) the probability that a given node will marry another
                node in people
        D: ((len(people) x len(people)) numpy array) indexed array of distance
            between nodes in people (siblings are distance 2)
        indices: (dictionary) maps node name (int) to index number in D (int)
    RETURNS:
        unions: (list of tuples of int) marriages formed.  Entries are of two
            types: 1) infite distance marraiges: one spouse is selected
            uniformly at random from the community (people) while the other is
            an immigrant to the community (IE a new node NOT listed in people).
            2) finite distance couples: both spouses are members of the
            community (IE listed in people).  These couples are selected at
            random according to the marriage_probs)
        num_immigrants: (int) the number of NEW people added to the graph.  As
            implemented ALL new people get married to someone in the current
            generation (herein people)
        marriage_distances: (list of int) one entry per marriage created during
            this function call (IE within this generation).  Each entry
            indicates the distance between spouses through their nearest common
            ancestor.  As before, a distnace of -1 indicates an infinite
            distance (IE one spouse immigrated into the community)
    """
    marriage_distances = []
    #print(people)
    people_set = set(people)  # for fast removal later
    # find the next 'name' to add to your set of people
    next_person = np.max(people) + 1
    # number of non-connected people to add
    num_immigrants = round(prob_marry_immigrant * len(people))  # m
    # marry off the immigrants at random to nodes in the current generation
    marry_strangers = np.random.choice(people, size=num_immigrants, replace=False)
    unions = {(spouse, immigrant) for spouse, immigrant
                                    in zip(marry_strangers,
                                           range(next_person, next_person + num_immigrants))}
    # remove the married people from the pool #2ndManifesto
    people_set = people_set - set(marry_strangers)
    # and record the (infinite) distances of each marriage
    marriage_distances += [-1 for k in range(num_immigrants)]

    # get number of people to marry
    num_couples_to_marry = round(len(people_set)*prob_marry/2)
    # get all possible pairs of the still single nodes
    # rejecting possible parrings which have a common ancestor more recently
    # than allowed by finite_marriage_probs (IE this is where we account that siblings
    # don't marry in most cultures (but still can such as in the tikopia_1930
    # family network))
    possible_couples = {(man, woman): D[indices[man]][indices[woman]]
                        for man, woman in itertools.combinations(people_set, 2)
                        if D[indices[man]][indices[woman]] > min(finite_marriage_probs)}
    iter = 0
    while possible_couples and iter < num_couples_to_marry:
        # find the probabilities of all possible distances
        # must update after each marriage
        # change to a data structure suited to random draws:
        # FIXME will this preserve ordering between keys and values?

        possible_couples_array = np.array(list(possible_couples.keys()))
        dis_probs = np.array([finite_marriage_probs[d] for d in possible_couples.values()])
        dis_probs[np.abs(dis_probs) < tol] = 0  # prevent "negative zeros"
        dis_probs = dis_probs / np.sum(dis_probs)  # normalize

        # choose couple based on relative probability of distances
        couple_index = np.random.choice(np.arange(len(possible_couples)), p=dis_probs)
        couple = possible_couples_array[couple_index]
        unions.add(tuple(couple))
        # and save the distance of that couple
        marriage_distances += [int(D[indices[couple[0]], indices[couple[1]]])]

        # remove all possible pairings which included either of the now-married couple
        possible_couples = {pair:possible_couples[pair]
                                for pair in possible_couples
                                    if ((pair[0] != couple[0])
                                    and (pair[1] != couple[0])
                                    and (pair[0] != couple[1])
                                    and (pair[1] != couple[1]))}
        iter += 1
        # test1 = [k for k in possible_couples if k[0] == couple[0]]
        # test2 = [k for k in possible_couples if k[0] == couple[1]]
        # test3 = [k for k in possible_couples if k[1] == couple[0]]
        # test4 = [k for k in possible_couples if k[1] == couple[1]]
        # if test1 or test2 or test3 or test4:
        #     print("missed some possibilities")
    return unions, num_immigrants, marriage_distances


def add_children_edges_kolton(unions, num_people, child_probs):
    """
    PARAMETERS:
        unions: (list of tuple of int) marriages in the current generation
            (output of kolton_add_marriage_edges())
        num_people: (int) current total number of nodes (persons) in the graph
            (IE the sum of the size of every generation)
        child_probs: (dictionary) keys are number of children (int), values are
            the probability (float) that a couple has key children (the output
            of get_probabilities(child_dist))
        indices: (dictionary) mapping the current generations' names (int) to
            index (row/column number) in the current distance matrix D
    RETURNS:
        child_edges: (list of tuple of int) entries are (parent, child) and
            should be added to the graph
        families: (list of list of int) entries are lists of children pertaining
            to the ith couple (follows the order of unions)
        num_people + total_num_children: (int) updated total number of people in
            the community/graph after adding children to the current generation
            of marriages
        num_children_each_couple: (np.array of len(unions) of int) each entry is
            a random draw from the child_probs distribution, how many children
            the ith couple of unions has
        indices: (dictionary) mapping the current generations' names (int) to
            index (row/column number) in the current distance matrix D, updated
            to include the new children (new children are added to D outside of
            this function)
    """
    families = []
    child_edges = []

    num_children_each_couple = np.random.choice(np.array(list(child_probs.keys())), p=np.array(list(child_probs.values())), size=len(unions))
    total_num_children = sum(num_children_each_couple)
    # names = [num_people + child for child in range(total_num_children)]
    # families = [[names[k]] for k in range(last + family ) for family in num_children_per_couple if
    # families = [[num_people + child for child in range(family)] for family in num_children_per_couple]
    # families = [[child for child in range(] for family in num_children_per_couple]
    biggest_name = num_people

    for union, num_children in zip(unions, num_children_each_couple):
        if num_children == 0:
            families.append([])
        else:
            children = [biggest_name + child for child in range(num_children)]
            biggest_name += num_children  # the next 'name' to use, next available index
            father_edges = [(union[0], child) for child in children]
            mother_edges = [(union[1], child) for child in children]
            child_edges.append(father_edges + mother_edges)
            families.append(children)
    # flatten the list of family edges
    child_edges = [edge for family in child_edges for edge in family]
    #max_ind = max(indices.values())
    #indices = indices | {child + num_people: ind for ind, child in zip(range(max_ind+1, max_ind+1+total_num_children), range(total_num_children))}
    return child_edges, families, num_people + total_num_children, num_children_each_couple


# sum(get_probabilities(marriage_dist).values())
# sum(get_probabilities(child_dist, is_child=True).values())
# sum(get_probabilities(child_dist).values())
# set(child_dist)


def update_distances_kolton(D, n, unions, families, indices):
    """
    Build a distance matrix that keeps track of how far away each node is from
    each other. Need to update distances after new nodes added to graph (i.e.,
    after adding children)
    PARAMETERS:
        D (array): "old" matrix of distances, previous generation
            n (int): number of nodes currently in graph
        unions: (list of tuple of int) marriages in the current
            generation (output of kolton_add_marriage_edges())
        no_unions: (list of int) list of nodes in the current generation
            which did not marry
        families: (list of list of int) entries are lists of children
            pertaining to the ith couple (follows the order of unions)
            (output of add_children_edges_kolton())
        indices (dictionary): maps node name (an int) to index number
            (row/column number) in the current distance matrix D.
    RETURNS:
        D1 (array): "new" (updated) matrix of distances
            for the current generation
        new_indices: (dictionary) mapping the current generations' names (int)
            to index (row/column number) in the current distance matrix D
    """
    # initialize new matrix
    num_children = len([child for fam in families for child in fam])
    D1 = np.zeros((num_children, num_children))
    new_indices = {child:k for k, child in enumerate([child for fam in families for child in fam])}

    # check_indices(new_indices)

    # compute new distances
    unions = list(unions)
    # unions
    # u, union = 0, (19, 43)
    # other = (21, 35)

    for u, union in enumerate(unions):
        u_children = families[u]

        for other in unions:
            if (union != other):
                o_children = families[unions.index(other)]

                # find all possible distances from union to other
                d1 = D[indices[union[0]]][indices[other[0]]]
                d2 = D[indices[union[1]]][indices[other[0]]]
                d3 = D[indices[union[0]]][indices[other[1]]]
                d4 = D[indices[union[1]]][indices[other[1]]]

                possible_distances = np.array([d1, d2, d3, d4])
                possible_distances = possible_distances[possible_distances > -1]  # IE where NOT infinite
                # compute distance between children of union and children of other
                d = np.min(possible_distances) + 2
                for uc in u_children:
                    for oc in o_children:
                        D1[new_indices[uc]][new_indices[oc]] = d
                        D1[new_indices[oc]][new_indices[uc]] = d

        # add immediate family distances
        for ch in u_children:
            # add sibling distances
            for c in u_children:
                if ch != c:
                    D1[new_indices[ch]][new_indices[c]] = 2
                    D1[new_indices[c]][new_indices[ch]] = 2
    return D1, new_indices


def check_indices(indices):
    all_ok = True
    min_child = min(indices.keys())
    max_child = max(indices.keys())
    for i in range(min_child, max_child-1,):
        if indices[i]+1 != indices[i+1]:
            all_ok = False
            print([(k, indices[k]) for k in range(i-2, i+2)])
    if all_ok:
        print("indices ok")


# TODO: what do we actually want this to return?
def human_family_network(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, children_dist, name, when_to_stop=np.inf, num_gens=np.inf, save=True):
    """
    PARAMETERS:
        num_people (int): number of people (nodes) to include in initial
            generation
        marriage_dists: (list of int) one entry per marriage indicating how many
            generations between spouses (reported in the number of parent-child
            edges crossed so that distance between siblings is 2) in the
            specified dataset.  If no common ancestor (IE an infinite distance)
            then the corresponding entry is -1
        prob_finite_marriage (float): probability of marriage being drawn from
            the finite portion of marriage_dist (herein defined and treated as
            finite_marriage_probs)
        prob_inf_marriage (float): probability of marriage to a non-connected person
        children_dist: (list of int) one entry per pair of parents, indicating
            how many child edges each parent in the couple share
        name: (str) name for prefix of saved files
        when_to_stop: (int) target number of nodes to capture.  If supplied, the
            model will run until this target number of nodes (all together, not
            just the size of the current generation) is surpassed.
        num_gens (int): max number of generations for network to grow beyond the
            initial generation.  Default is np.inf.  If np.inf, then the model
            will run until the number of nodes in the example network is
            surpassed and then stop.

    RETURNS:
        G:
        D:
        unions:
        num_children_per_couple:
    """
    dies_out = False

    all_marriage_edges = []
    all_marriage_distances = []
    all_children_per_couple = []

    G = nx.DiGraph()
    # num_finite_dist = round(num_people * prob_finite_marriage)
    # num_inf_dist = round(num_people * prob_inf_marriage)
    # num_single = num_people - num_finite_dist - num_inf_dist
    marriage_dist_array = np.array(marriage_dist)
    finite_only_marriage_dist = marriage_dist_array[marriage_dist_array != -1]
    d = np.triu(np.random.choice(finite_only_marriage_dist, size=(num_people, num_people)), k=1)
    D = d + d.T
    indices = {node:k for k, node in enumerate(range(num_people))}
    generation_of_people = list(indices.values())
    # explicitly add our first generation of nodes (otherwise we will fail to
    # add those who do not marry into our graph).  All future generations are
    # connected either through marriage or through parent-child arcs
    G.add_nodes_from(generation_of_people)
    # get probabilities of possible finite distances to use in marriage function
    # and normalize it
    finite_marriage_probs = get_probabilities(marriage_dist)
    finite_marriage_probs = {key:value/sum(finite_marriage_probs.values()) for key, value in zip(finite_marriage_probs.keys(), finite_marriage_probs.values())}
    # now make the finite marriage entries sum to the probability of a finite marriage
    finite_marriage_probs = {key:value*prob_finite_marriage for key, value in zip(finite_marriage_probs.keys(), finite_marriage_probs.values())}
    # now add an entry for infinite distance marriages
    # ??? TODO should we add the infinite distance portion the way rebbekah did?
    # marriage_probs[100] = (infdis/len(all_distances))/2  # include probability of infinite distance
    # factor = 1.0/sum(marriage_probs.values())   # normalizing factor
    # # normalize values for finite and infinite distances
    # for k in marriage_probs:
    #     marriage_probs[k] = marriage_probs[k]*factor
    # marriage_probs = finite_marriage_probs.copy()
    # marriage_probs[-1] = prob_inf_marriage
    # and add an entry for staying single
    # marriage_probs[0] = 1 - prob_finite_marriage - prob_inf_marriage

    # ditto for the child distribution
    child_probs = get_probabilities(children_dist, is_child=True)
    # ??? make probabilities non-negative (some entries are effectively zero, but negative)
    child_probs = {key:value if value > 0 else 0 for key, value in zip(child_probs.keys(), child_probs.values()) }
    child_probs = {key:value/sum(child_probs.values()) for key, value in zip(child_probs.keys(), child_probs.values())}

    # add specified number of generations to the network
    print("generation 0: num_people ", num_people)
    i = 1
    while (num_people < when_to_stop) & (i < num_gens):
        # check_indices(indices)

        # for i in range(num_gens):
        # create unions between nodes to create next generation
        #unions, no_unions, all_unions, n, m, infdis, indices = add_marriage_edges(all_fam, all_unions, D, marriage_probs, p, ncp, n, infdis, indices)
        if len(generation_of_people) == 0:
            dies_out = True
            break
        unions, num_immigrants, marriage_distances = kolton_add_marriage_edges(generation_of_people, finite_marriage_probs, prob_inf_marriage, prob_finite_marriage, D, indices)
        # marriage edges should be undirected
        # ??? TODO should we add both directions in?  We will be able to grab just "Marriage" edges and then undirect this graph equivalently
        G.add_edges_from(unions, Relationship="Marriage")
        all_marriage_edges += list(unions)
        all_marriage_distances += marriage_distances

        for j in range(num_immigrants):
            # add non-connected people to distance matrix
            r = np.ones((1, len(generation_of_people) + 1 + j)) * -1  # -1 is infinite distance
            r[0, -1] = 0  # distance to self is 0
            c = np.ones((len(generation_of_people) + j, 1)) * -1  # -1 is infinite distance
            D = np.hstack((D, c))
            D = np.vstack((D, r))

        max_ind = max(indices.values())
        indices = indices | {immigrant + num_people:ind for ind, immigrant in zip(range(max_ind + 1, max_ind+1+num_immigrants), range(num_immigrants))}
        num_people += num_immigrants

        # print('A')
        # check_indices(indices)
        # D0A = D.copy()
        # indices0A = indices.copy()
        # num_people0A = num_people
        #
        # D = D0A.copy()
        # indices = indices0A.copy()
        # num_people = num_people0A
        # D.shape
        # len(indices)

        # add children to each marriage
        child_edges, families, num_people, num_children_per_couple = add_children_edges_kolton(unions, num_people, child_probs)
        G.add_edges_from(child_edges, Relationship='Parent-Child')
        all_children_per_couple += list(num_children_per_couple)

        # print('B')
        # check_indices(indices)
        # D0B = D.copy()
        # indices0B = indices.copy()
        #
        # D = D0B.copy()
        # indices = indices0B.copy()
        # D.shape
        # len(indices)

        # update distances between nodes
        D, indices = update_distances_kolton(D, num_people, unions, families, indices)

        # print('C')
        # check_indices(indices)
        # D0C = D.copy()
        # indices0C = indices.copy()
        # D = D0C.copy()
        # indices =indices0C.copy()

        generation_of_people = list(indices.keys())
        print('generation: ', i, '  num_children: ', len(generation_of_people), '   num_immigrants: ', num_immigrants, '   total people: ', num_people)
        i += 1
        print('number of nodes in graph: ', G.number_of_nodes())
        # ??? save output at each generation
    if save:
        output_path = makeOutputDirectory("output", name)
        Gname = Gname = "{}/{}_G.gpickle".format(output_path, name)   # save graph
        nx.write_gpickle(G, Gname)
        Uname = "{}/{}_marriage_edges".format(output_path, name)   # save unions
        with open(Uname, 'wb') as fup:
            pickle.dump(all_marriage_edges, fup)
        Dname = "{}/{}_marriage_distances".format(output_path, name) # save marriage distances
        with open(Dname, 'wb') as myfile:
            pickle.dump(all_marriage_distances, myfile)
        Cname = "{}/{}_children_per_couple".format(output_path, name)   # save children
        with open(Cname, 'wb') as fcp:
            pickle.dump(all_children_per_couple, fcp)

        # # save output of the last generation
        # if i == gen:
        #     print("Last generation: ", i+1)
        #     Gname = "{}_G{}.gpickle".format(name, i+1)   # save graph
        #     nx.write_gpickle(G, Gname)
        #     Dname = "{}_D{}.npy".format(name, i+1)   # save D
        #     np.save(Dname, D)
        #     indicesname = "{}_indices{}.npy".format(name, i)  # save indices
        #     np.save(indicesname, indices)
        #     Uname = "{}_U{}".format(name, i+1)   # save unions
        #     with open(Uname, 'wb') as fup:
        #         pickle.dump(all_unions, fup)
        #     Cname = "{}_C{}".format(name, i+1)   # save children
        #     with open(Cname, 'wb') as fcp:
        #         pickle.dump(all_children, fcp)

    return G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out


"""
below is example code to run the model
"""
#
# name = 'tikopia_1930'
# # name = 'achuar_huasaga_chankuap'
# num_people = 50
#
# # num_gens = 20
# marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
# G, all_marriage_edges, all_marriage_distances, all_children_per_couple = human_family_network(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, child_dist, name, when_to_stop=size_goal)

# size_goal
# G.number_of_nodes()


# csv with name of graph and starting nodes
# The goal of this function is to find the "right" number of starting nodes.
# def find_start_size(name, max_iters=100): # n = number of initial nodes
#     iter = 0
#
#     marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
#     num_people = size_goal//2
#     dies_out = 0 # counter for the number of times the model dies out
#     #     dies_out_lower_bound = 40 # the lower bound of how many times the model should die out
#     #     dies_out_upper_bound = 60 # the upper bound of how many times the model should die out
#     dies_out_threshold = 5
#     greatest_lower_bound = 2
#     least_upper_bound = size_goal
#     while dies_out != dies_out_threshold and iter < max_iters: # while the number of times the model dies out is not equal to the threshold of dying:
#         iter += 1
#
#         G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies = human_family_network(num_people,
#                                                                                                             marriage_dist,
#                                                                                                             prob_finite_marriage,
#                                                                                                             prob_inf_marriage,
#                                                                                                             child_dist,
#                                                                                                             name,
#                                                                                                             when_to_stop=size_goal,
#                                                                                                             save=False)
#         if dies:
#             dies_out += 1
#         # I would get some value for dies_out
#
#         if dies_out > dies_out_threshold:  # we want to increase num_people
#             greatest_lower_bound = num_people  # current iteration died out too frequently.  Won't need to search below this point again.
#             num_people = (num_people + least_upper_bound) // 2 # midpoint between num_people and size_goal
#             dies_out = 0
#             iter = 0
#         elif dies_out < dies_out_threshold and iter == max_iters: # we want to decrease num_people
#             least_upper_bound = num_people  # current iteration died out too infrequently.  Won't need to search above this point again
#             num_people = (greatest_lower_bound + num_people) // 2 # midpoint between 2 and num_people
#             dies_out = 0
#             iter = 0
#         print('starting population: ', num_people)
#
#     print("Size goal:", size_goal)
#     print("Number of nodes:", G.number_of_nodes())
#     return num_people


def find_start_size(name, max_iters=100): # n = number of initial nodes
    iter = 0

    marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
    num_people = size_goal//2
    dies_out = 0 # counter for the number of times the model dies out
    #     dies_out_lower_bound = 40 # the lower bound of how many times the model should die out
    #     dies_out_upper_bound = 60 # the upper bound of how many times the model should die out
    dies_out_threshold = 5
    greatest_lower_bound = 2
    least_upper_bound = size_goal
    while dies_out != dies_out_threshold: # while the number of times the model dies out is not equal to the threshold of dying:

        for i in range(max_iters):
            G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies = human_family_network(num_people,
                                                                                                                marriage_dist,
                                                                                                                prob_finite_marriage,
                                                                                                                prob_inf_marriage,
                                                                                                                child_dist,
                                                                                                                name,
                                                                                                                when_to_stop=size_goal,
                                                                                                                save=False)
            if dies:
                dies_out += 1
            if dies_out > dies_out_threshold:
                break
        print('greatest_lower_bound: ', greatest_lower_bound)
        print('least_upper_bound: ', least_upper_bound)
        if greatest_lower_bound >= least_upper_bound - 1:
            # IE the ideal lies between these two integers
            # so return the larger
            num_people = least_upper_bound
            break
        elif dies_out == dies_out_threshold:
            break
        elif dies_out > dies_out_threshold:  # we want to increase num_people
            greatest_lower_bound = num_people  # current iteration died out too frequently.  Won't need to search below this point again.
            num_people = (num_people + least_upper_bound) // 2 # midpoint between num_people and size_goal
            dies_out = 0
            iter = 0
        elif dies_out < dies_out_threshold: # we want to decrease num_people
            least_upper_bound = num_people  # current iteration died out too infrequently.  Won't need to search above this point again
            num_people = (greatest_lower_bound + num_people) // 2 # midpoint between 2 and num_people
            dies_out = 0
            iter = 0
        print('starting population: ', num_people)

# find_start_size(name, max_iters=100)
# print('done')
#
# marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
# num_people = 66
# dies_out = 0 # counter for the number of times the model dies out
# #     dies_out_lower_bound = 40 # the lower bound of how many times the model should die out
# #     dies_out_upper_bound = 60 # the upper bound of how many times the model should die out
# dies_out_threshold = 5
# greatest_lower_bound = 2
# least_upper_bound = size_goal
# max_iters = 1000
# for i in range(max_iters):
#     G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies = human_family_network(num_people,
#                                                                                                         marriage_dist,
#                                                                                                         prob_finite_marriage,
#                                                                                                         prob_inf_marriage,
#                                                                                                         child_dist,
#                                                                                                         name,
#                                                                                                         when_to_stop=size_goal,
#                                                                                                         save=False)
#     if dies:
#         dies_out += 1
# dies_out
