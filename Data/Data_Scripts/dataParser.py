import os
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx 

######## --------- FUNCTIONS USED TO PARSE .paj FILES --------- ##########
def get_paj_file_names(path = os.path.join(os.path.dirname(__file__),'../Original_Sources'), sort = True):
    """Creates list of the names of the .paj files in a given directory
    
    Parameters:
        path (str): The file path to the folder to search for .paj files in.
        sort (boolean): Whether or not to sort the names alphabetically 
            (default is True)

    Returns:
        paj_file_names (list): A list of names of .paj files in that directory
    """
    # Initialize a list to hold the names of the .paj files
    paj_file_names = []

    # Walk through the directory specified by 'path'
    abspath = os.path.abspath(path)
    for root, dirs, files in os.walk(abspath):
        # Get all of the files in that directory
        for file in files:
            # If and only if this is a .paj file, 
            # record its name in graph_names.
            if os.path.splitext(file)[1] == '.paj':
                paj_file_names.append(file)

    # If 'sorted' is True, sort these names alphabetically
    if sort:
        sorted_names_indexes = np.argsort(paj_file_names)
        paj_file_names = [paj_file_names[i] for i in sorted_names_indexes]
    
    # Return the list of .paj file names
    return paj_file_names

def get_children_tuples(paj_file_name, paj_file_directory= os.path.join(os.path.dirname(__file__),'../Original_Sources')):
    """Gets all of the children tuples found in the *arcs section of a .paj file

    Parameters:
        paj_file_name (str): The name of the .paj file to parse
        paj_file_directory (str): The relative file path to the folder where 
            the .paj file is stored
   
    Returns:
        children (list): A list of tuples containing two-tuples of the 
            form (parent ID, child ID)
    """
    # Read each line of the .paj file into its own entry in a list
    paj_file = os.path.join(paj_file_directory, paj_file_name)
    with open(paj_file, 'r') as file:
        paj_file_contents = file.readlines()
    
    # Get all of the children tuples from the *arcs section of the file 
    children = []
    start_children = False 

    # Iterate through each line of the .paj file
    for item in paj_file_contents:
        # Figure out if we have found *arcs or *edges 
        # (which signal the start/end of the children section)
        if item.find('*arcs') != -1:
            start_children = True 
        elif item.find('*edges') != -1:
            break # Means the end of the children section
        elif start_children:
            edge_info = [int(i) for i in item.split()]
            parent = edge_info[0]  # First number in tuple is parent's ID
            child = edge_info[1]   # Second number in tuple is child's ID 
            children.append((parent, child)) # Append the edge (parent, child) to the list
    return children

def get_marriage_tuples(paj_file_name, paj_file_directory = os.path.join(os.path.dirname(__file__),'../Original_Sources')):
    """Gets all of the marriage tuples found in the *edges section of a .paj file

    Parameters:
        paj_file_name (str): The name of the .paj file to parse
        paj_file_directory (str): The relative file path to the folder where 
            the .paj file is stored
   
    Returns:
        marriages (list): A list of tuples containing two-tuples of the 
            form (spouse 1 ID, spouse 2 ID)
    """
    # Read each line of the .paj file into its own entry in a list
    paj_file = os.path.join(paj_file_directory, paj_file_name)
    with open(paj_file, 'r') as file:
        paj_file_contents = file.readlines()
    
    # Get all of the children tuples from the *edges section of the file 
    marriages = []
    start_marriages = False 

    # Iterate through each line of the .paj file
    for item in paj_file_contents:
        # Figure out if we have found *edges 
        # (which signal a start to the marriage section)
        if item.find('*edges') != -1:
            start_marriages = True 
        elif start_marriages:
            edge_info = [int(i) for i in item.split()]
            first_partner = edge_info[0]  
            second_partner = edge_info[1]
            # Append the marriage edge to the marriage list 
            marriages.append((first_partner, second_partner)) 
    
    return marriages

######## --------- FUNCTIONS USED TO CREATE MARRIAGE/CHILDREN DISTRIBUTIONS --------- ##########
def get_children_distributions(children_tuples, marriage_tuples):
    """Get the number of children in every family.
    
    Parameters:
        children_tuples (list): A list of tuples of the form (parent ID, child ID)
        marriage_tuples (list): A list of tupels of the form (spouse1 ID, spouse2 ID)
    
    Returns:
        complete_dist (list): A list, with entries being the number of children in each 
                              family (all categories included)
        married_parent_dist (list): A list, with entries being the number of children in each
                                    family with married parents
        unmarried_parent_dist (list): A list, with entries being the number of children in each
                                      family with unmarried parents
        single_parent_dist (list): A list, with entries being the number of children in each
                                   family with a single parent
    """    
    #### 1) Populate lookup dictionaries for family information
    married_fams = {}           # A lookup dictionary for families with married parents
    unmarried_fams= {}          # A lookup dictionary for families with unmarried parents
    single_parent_fams = {}     # A lookup dictionary for families with a single parent

    #### 2) Initialize the married-family lookup dictionary with all married edge tuples as keys
    ####    with the format {(parent1, parent2): {set of unique children}}
    for parent_pair in marriage_tuples:
        married_fams[parent_pair] = set()

    #### 3) Get parent/spouse info for each person in the network
    relationship_info_dict = get_relationship_info(children_tuples, marriage_tuples)

    #### 4) Add each child to the appropriate family in the appropriate lookup dictionary
    for person in relationship_info_dict.keys():
        parents_list = relationship_info_dict[person]['Parents']
        if len(parents_list) == 0:      # No parent info - should not be counted as a child
            continue
        elif len(parents_list) == 1:    # Only 1 parent found - counted as child of SINGLE PARENT 
            parent1 = parents_list[0]
            if parent1 not in single_parent_fams.keys():
                single_parent_fams[parent1] = set()
            single_parent_fams[parent1].add(person)
        else:                            # 2 parents found - remains to be seen if the parents are married
            found = False
            parent1 = parents_list[0]
            parent2 = parents_list[1]
            for couple in married_fams.keys():
                candidate1 = couple[0]
                candidate2 = couple[1]
                if (candidate1==parent1 or candidate1==parent2) and (candidate2==parent1 or candidate2==parent2):
                    # If we get here, the child's parents are MARRIED
                    married_fams[couple].add(person)
                    found = True
                    break 
            if not found:  # Child's parents were not in *edges - they are a child of an orphaned couple.
                # if we get here, the child's parents are UNMARRIED
                unmarried_found = False
                for couple in unmarried_fams.keys():
                    candidate1 = couple[0]
                    candidate2 = couple[1]
                    if (candidate1==parent1 or candidate1==parent2) and (candidate2==parent1 or candidate2==parent2):
                        # We found the unmarried parents already - just add the child to their family
                        unmarried_fams[couple].add(person)
                        unmarried_found = True
                        break 
                if not unmarried_found:
                    unfound_parents = (parent1, parent2)
                    unmarried_fams[unfound_parents] = {person}
    
    #### 5) Calculate the number of children in each family to create the distribution 
    single_parent_dist = get_family_children_distributions(single_parent_fams)
    married_parent_dist = get_family_children_distributions(married_fams)
    unmarried_parent_dist = get_family_children_distributions(unmarried_fams)

    complete_dist = []
    complete_dist.extend(single_parent_dist)
    complete_dist.extend(married_parent_dist)
    complete_dist.extend(unmarried_parent_dist)

    return complete_dist, married_parent_dist, unmarried_parent_dist, single_parent_dist

def get_family_children_distributions(family_info):
    """Constructs a distribution of the number of children per family.
    
    Parameters:
        family_info (dict): A dictionary of the format {parent ID/tuple of IDs: {set of unique children IDs}
    
    Returns:
        children_distribution (list): A list of the number of children in each family
    """
    # Create a list to hold the number of unique children in each family
    children_distribution = []

    total_num_children = 0
    for family in family_info.keys():
        # Append the number of unique children in each family to the children_distribution list
        children_ids = family_info[family]
        num_children = len(children_ids)
        children_distribution.append(num_children)
        total_num_children += num_children 
    
    # Return the distribution of the number of children in each family
    return children_distribution

def get_relationship_info(children, marriages):
    """Populate a dictionary of information about each person in a genealogical network
    (including parent and spouse information) from the contents of a .paj file

    Parameters:
        children (list): A list of tuples of the form (parent ID, child ID)
        marriages (list): A list of tupels of the form (spouse1 ID, spouse2 ID)

    Returns:
        graph_info : A dictionary of the form 
        {personID:{spouses:list of spouse IDs, parents:list of parent IDs}}
    """
    # Create a dictionary of information for each unique person ID in the network 
    graph_info = {}
    
    # Populate parent-child information from the child tuples
    for child_tuple in children:
        parent = child_tuple[0]
        child = child_tuple[1]
        if child not in graph_info.keys():
            graph_info[child] = {"Spouses":[], "Parents":[]}
        graph_info[child]["Parents"].append(parent)
    
    # Populate spouse information from the marriage tuples 
    for marriage_tuple in marriages:
        spouse1 = marriage_tuple[0]
        spouse2 = marriage_tuple[1]
        if spouse1 not in graph_info.keys():
            graph_info[spouse1] = {"Spouses":[], "Parents":[]}
        if spouse2 not in graph_info.keys():
            graph_info[spouse2] = {"Spouses":[], "Parents":[]}
        graph_info[spouse1]["Spouses"].append(spouse2)
        graph_info[spouse2]["Spouses"].append(spouse1)
    
    # Return the information dictionary created for this network
    return graph_info


###### FUNCTIONS USED TO CREATE/ANALYZE NETWORK STRUCTURES ############
# TODO FINISH THESE
def create_children_network(children_tuples):
    """Creates a Networkx Graph from the children tuples
    
    Parameters:
        children_tuples (list): A list of tuples of the form 
            (parent_id, child_id)
    
    Returns:
        children_network (nx.Graph): A Networkx graph representing
        the parent/child relationships represented by the tuples.
    """
    # Initialize an empty graph
    children_network = nx.Graph()

    # Add each tuple relationship to the graph as an *UNDIRECTED* edge 
    # TODO should this be directed, or undirected?
    for relationship in children_tuples:
        parent = relationship[0]
        child = relationship[1]
        children_network.add_edge(parent, child)
        children_network.add_edge(child, parent)
    
    # Return the resulting network
    return children_network

def plot_degree_distribution(graph, graph_title):
    """Plots the degree distribution of a graph as a histogram.
    
    Parameters:
        graph (nx.Graph): The graph to analyze
        graph_title (str): What the graph should be titled.

    Returns:
        degree_seq (list): The degree sequence of the graph
    """
    # Get a list of the NUMBER of nodes with each degree
    graph_degree_summary = nx.degree_histogram(graph)

    # Iterate over the degree summary to create the actual degree
    # sequence for the graph (for ease of plotting)
    degree_seq = []
    for degree,num_nodes in enumerate(graph_degree_summary):
        for _ in range(num_nodes):
            degree_seq.append(degree)
    
    # Show the histogram representing the degree distribution
    plt.hist(degree_seq)
    plt.title(graph_title)
    plt.show()

    # Return the degree sequence 
    return degree_seq
    

####### DRIVER TO GET/OUTPUT ALL DISTRIBUTIONS ######### 
def main():
    """Gather and output the children distributions for all original data sources"""
    #### 1) Get the name of each .paj file in Original_Sources 
    ####    (sorted in alphabetical order)
    print("Getting children distributions . . .")
    sorted_graph_names = get_paj_file_names()

    #### 2) Iterate over each of these .paj files, and parse their data
    for graph_name in sorted_graph_names:
        ### 2a) Get the marriage/parent-child tuples from the .paj file
        children_tuple_list = get_children_tuples(graph_name)
        marriage_tuple_list = get_marriage_tuples(graph_name)

        ### 2b) Get the distributions for the number of children per family 
        ###     (in a variety of different family circumstances)
        distributions = get_children_distributions(children_tuple_list, marriage_tuple_list)
        complete_dist = distributions[0]
        married_parent_dist = distributions[1]
        unmarried_parent_dist = distributions[2]
        single_parent_dist = distributions[3]

        ### 2c) Output each of these distributions as a string to a .txt file
        ###     for further parsing 
        OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__),'../Parsed_Data/Children_Distributions')
        name_without_ext = os.path.splitext(graph_name)[0]
        name_without_ext = "-".join(name_without_ext.split("-")[1:-1])

        complete_dist_file = os.path.join(OUTPUT_DIRECTORY, f'Complete_Distributions/{name_without_ext}.txt')
        with open(complete_dist_file, 'w') as out_file:
            out_file.write(str(complete_dist))

        married_dist_file = os.path.join(OUTPUT_DIRECTORY, f'Married_Couple_Distributions/{name_without_ext}.txt')
        with open(married_dist_file, 'w') as out_file:
            out_file.write(str(married_parent_dist))
        
        unmarried_dist_file = os.path.join(OUTPUT_DIRECTORY, f'Unmarried_Couple_Distributions/{name_without_ext}.txt')
        with open(unmarried_dist_file, 'w') as out_file:
            out_file.write(str(unmarried_parent_dist))
        
        single_dist_file = os.path.join(OUTPUT_DIRECTORY, f'Single_Parent_Distributions/{name_without_ext}.txt')
        with open(single_dist_file, 'w') as out_file:
            out_file.write(str(single_parent_dist))

    print("Done.")

if __name__ == "__main__":
    main()
