import pickle
from pathlib import Path
import os
import networkx as nx
from metis import part_graph
from tqdm import tqdm
import math
import numpy as np
import dwave

from collections import defaultdict
from dimod import SampleSet

def solution_correction(
        stable_set_graph,
        solution_nodes
    ):
    subgraph = nx.Graph(stable_set_graph.subgraph(solution_nodes))
    # we go through each vertex, and if it has any edges connected to it, we remove it.
    for n in list(subgraph.nodes):
        if(subgraph.degree(n) > 0):
            subgraph.remove_node(n)
    return list(subgraph.nodes)

def calculate_partition_with_halo(
        stable_set_graph,
        num_of_part
    ):
    clique_graph = nx.complement(stable_set_graph)
    objval, metis_partition_list = part_graph(clique_graph, nparts=num_of_part)
    # metis_partition_list (1 dimensional list) is mapping from vertex i to partition number metis_partition_list[i].
    # this means that number of partitions is max number in it + 1
    num_partitions = max(metis_partition_list)+1

    partition_with_halo_list = [None]*num_partitions
    partition_list = [None]*num_partitions
    # our partition format will be list of sets, where each set will contain vertice in partition
    # the below code coverts metis format to ours
    node_list = list(clique_graph.nodes)
    for i in range(num_partitions):
        partition_list[i] = set([node_list[j] for j, value in enumerate(metis_partition_list) if value == i])

    # to find the partition together with the halo, we go through neighbourhood of every vertex in the partition and we
    # make an union of each such neighbourhood with the current partition_list.
    for i in range(num_partitions):
        partition_with_halo_list[i] = set()
        partition_with_halo_list[i] = partition_with_halo_list[i].union(partition_list[i])
        for node in partition_list[i]:
            partition_with_halo_list[i] = partition_with_halo_list[i].union(set(clique_graph[node].keys()))
    
    return partition_with_halo_list

def print_best_partitions(
        stable_set_graph,
        max_num_of_part,
        sort='v'
    ):

    results = []
    # we calculate partition for every number of partition in desired range, we map the result so that we are
    # left with tuples (# of partitions, max # of vertices in partition, max # of edges in partition), and 
    # we sort the tuple according to desired criteria
    for num_of_part in tqdm(range(2, max_num_of_part+1)):
        partition_with_halo_list = calculate_partition_with_halo(stable_set_graph, num_of_part)
        results.append((num_of_part,
                    max(list(map(lambda x : len(x), partition_with_halo_list))),
                    max(list(map(lambda x : len(stable_set_graph.subgraph(x).edges()), partition_with_halo_list)))
        ))
    if(sort == 'v'):
        results.sort(key = lambda x : x[1])
    elif(sort == 'e'):
        results.sort(key = lambda x : x[2])
    
    # This is just a lot of effort to make the output aligned
    part_pad = max(list(map(lambda x : int(math.log(x[0], 10)), results)))+1
    vertex_pad = max(list(map(lambda x : int(math.log(x[1], 10)), results)))+1
    edge_pad = max(list(map(lambda x : int(math.log(x[2], 10)), results)))+1
    for i in reversed(results):
        a = f"{i[0]:>{part_pad}}"; b = f"{i[1]:>{vertex_pad}}"; c = f"{i[2]:>{edge_pad}}"
        print(f"Number of partitions: {a} || Max number of vertices: {b} || Max number of edges {c}")

def calculate_best_solution(
        stable_set_graph,
        sampler,
        beta=1,
        num_of_runs=1,
        num_of_part=1,
        output_file=None,
        no_output_file = False,
        console_output=True,
        partition_with_halo_list=None
    ):
    # if we want to produce a file but the directory does not exist or is not given, throw an Exception,
    # otherwise we might waste computational resources and be left without the result
    if(not no_output_file and (output_file is None or not os.path.exists(os.path.dirname(output_file)))):
        raise Exception("Output directory not given or does not exist! If you don't want to produce the output file, set no_output_file=True in function call.")
    if(False):
        print(f"Using {sampler} sampler")
    # this has to be done, otherwise the partition function with input num_of_part=1 throws an error
    if(num_of_part > 1 and partition_with_halo_list is None):
        partition_with_halo_list = calculate_partition_with_halo(stable_set_graph, num_of_part=num_of_part)
    elif(partition_with_halo_list is None):
        partition_with_halo_list = [set(stable_set_graph.nodes)]

    best_solution_nodes = []
    best_solution_energy = 0
    all_solutions_info = {}

    # we go through every partition, we create subgraph and Q matrix (which represents QUBO function)
    # we calculate the response, which is saved in a file (if desired), we save the best solution nodes
    # and energy, and we print the best result
    for k, partition in enumerate(partition_with_halo_list):
        if(len(partition_with_halo_list) > 1):
            subgraph = stable_set_graph.subgraph(list(partition))
        else:
            subgraph = stable_set_graph

        Q = defaultdict(int)
        for i in partition:
            Q[(i,i)]+= -1
        for i, j in subgraph.edges:
            Q[(i,j)]+= beta

        
        if(num_of_runs > 1):
            response = sampler.sample_qubo(Q, num_reads=num_of_runs)
        else:
            response = sampler.sample_qubo(Q)

        if(not no_output_file):
            if(len(partition_with_halo_list) > 1):
                with open(output_file+"_partition"+str(k).zfill(int(math.log10(len(partition_with_halo_list)-1)+1))+".pkl", 'wb') as file:
                    pickle.dump(response.to_serializable(), file)
            else:
                with open(output_file, 'wb') as file:
                    pickle.dump(response.to_serializable(), file)

        # this is how you get solution nodes and solution energy from the response
        solution_nodes = [h for h,v in response.first[0].items() if v == 1]
        corrected_solution_nodes = solution_correction(stable_set_graph, solution_nodes)
        solution_energy = response.first[1]

        all_solutions_info[k] = {
            "best_solution_nodes" : corrected_solution_nodes,
            "best_solution_energy" : (-1)*len(corrected_solution_nodes),
            "best_uncorrected_solution_nodes": solution_nodes,
            "best_uncorrected_solution_energy" : solution_energy,
            "sample_set": response
        }
        
        if(solution_energy < best_solution_energy):
            best_solution_nodes = solution_nodes
            best_solution_energy = solution_energy
        
        # this is to avoid double printing the result if we choose to calculate response without partitions
        if(len(partition_with_halo_list) > 1 and console_output):
            k_pad = f"{k+1:>{int(math.log(len(partition_with_halo_list),10))+1}}"
            print(f"Partition {k_pad}: beta={beta} || solution energy: {solution_energy} || number of edges in solution: {len(subgraph.subgraph(solution_nodes).edges)} || solution size: {len(solution_nodes)}")

    if(console_output):
        print(f"Best result: beta={beta} || best solution energy: {best_solution_energy} || number of edges in best solution: {len(subgraph.subgraph(best_solution_nodes).edges)} || best solution size: {len(best_solution_nodes)}\n")

    return all_solutions_info

def open_saved_sample_set(input_file):
    with open(input_file, 'rb') as file:
        sample_set = SampleSet.from_serializable(pickle.load(file))
    return sample_set

# This function relies heavily on the theory from the document
def eliminate_and_recalculate(
        stable_set_graph,
        sample_set,
        sampler=None,
        num_of_runs=1
    ):
    # Calculate gamma as defined in the proposition 4.1
    gamma = int((-1)*sample_set.first[1])
    potential_solutions = []
    first_solution = [k for k,v in sample_set.first[0].items() if v == 1]
    # If first solution will not get added in the loop, add it now. We are doing this to avoid duplicates 
    if(len(first_solution)-1 <= gamma):
        potential_solutions.append(first_solution)
    for record in sample_set.record:
        record_nodes = [sample_set.variables[k[0]] for k,v in np.ndenumerate(record[0]) if v == 1]
        record_energy = (-1)*record[1]
        record_number_of_edges = len(stable_set_graph.subgraph(record_nodes).edges)
        # If we can't eliminate the node using point 2 and 3 from proposition 4.2, we add it
        if(record_number_of_edges != 0 and record_energy + record_number_of_edges - 1 > gamma):
            potential_solutions.append(record_nodes)

    recalculated_solutions = []
    # Recalcualte all solutions that we couldn't eliminate
    for solution in tqdm(potential_solutions):
        stable_set_subgraph = stable_set_graph.subgraph(solution)
        # If sampler is given, use it, otherwise just use the default algorithm described in the proof of proposition 4.1
        if(sampler is not None):
            sample_set = calculate_best_solution(stable_set_subgraph, sampler, num_of_runs=num_of_runs, no_output_file=True, console_output=False)
        else:
            corrected_solution = solution_correction(stable_set_graph, solution)
            sample_set = [{"best_solution_nodes": corrected_solution, "best_solution_energy": (-1)*len(corrected_solution)}]    
        recalculated_solutions.append({"recalculated_solution_nodes": sample_set[0]["best_solution_nodes"],
                                       "recalcualted_solution_energy": sample_set[0]["best_solution_energy"]})
    
    return {"best_recalculated_solution": min(recalculated_solutions, key= lambda x : x["recalcualted_solution_energy"]),
            "all_recalculated_solutions": recalculated_solutions}

def k_edge_core(clique_graph, k):
    # This is to make sure that the loop condition is always true the first time
    edges_before = len(clique_graph.edges)+1
    edges_after = len(clique_graph.edges)
    while(edges_before > edges_after):
        edges_before = edges_after
        for edge in clique_graph.edges:
            if(len(list(nx.common_neighbors(clique_graph, *edge))) < k):
                clique_graph.remove_edge(*edge)
        edges_after = len(clique_graph.edges)
    return clique_graph

def k_reduce_graph(clique_graph, k):
    nodes_before = len(clique_graph.nodes)+1
    nodes_after = len(clique_graph.nodes)
    while(nodes_before > nodes_after or edges_before > edges_after):
        nodes_before = nodes_after
        k_core_clique_graph = nx.k_core(clique_graph, k-1)
        nodes_after = len(k_core_clique_graph.nodes)

        edges_before = len(k_core_clique_graph.edges)
        k_edge_core_clique_graph = k_edge_core(k_core_clique_graph, k-2)
        edges_after = len(k_edge_core_clique_graph.edges)

        clique_graph = k_core_clique_graph
    return clique_graph

def first_k_for_reduction(stable_set_graph):
    clique_graph = nx.complement(stable_set_graph)
    min_common_neighbors = len(stable_set_graph.nodes)
    min_degree = min(dict(clique_graph.degree()).values())
    for edge in clique_graph.edges:
        min_common_neighbors = min(len(list(nx.common_neighbors(clique_graph, *edge))), min_common_neighbors)
    return min(min_degree, min_common_neighbors)

def find_k_core_upper_bound(stable_set_graph):
    clique_graph = nx.complement(stable_set_graph)
    k_min = min(
                min(dict(clique_graph.degree()).values()),
                min(list(map(lambda x : len(list(nx.common_neighbors(clique_graph, *x))), clique_graph.edges)))
            )
    k_max = max(
                max(dict(clique_graph.degree()).values()),
                max(list(map(lambda x : len(list(nx.common_neighbors(clique_graph, *x))), clique_graph.edges)))
            ) + 1
    while k_min < k_max-1:
        k_mid =(k_min+k_max) // 2
        reduced_graph = k_reduce_graph(clique_graph, k_mid)
        if(len(reduced_graph.nodes) < k_mid):
            k_max = k_mid
        else: 
            k_min = k_mid
    return k_min

#=================================================================================
def k_d_reduce_upper_bound(stable_set_graph, d, k_min=1):
    clique_graph = nx.complement(stable_set_graph)
    # Handeling special case when graph doesn't have edges, otherwise we get an error
    if(len(clique_graph.edges) == 0):
        return len(clique_graph.nodes)
    k_max = max(
                max(dict(clique_graph.degree()).values()),
                max(list(map(lambda x : len(list(nx.common_neighbors(clique_graph, *x))), clique_graph.edges)))
            ) + 1
    while k_min < k_max-1:
        k_mid =(k_min+k_max) // 2
        reduced_graph = k_d_reduce_graph(clique_graph, k_mid, d=min(d, k_mid))
        if(len(reduced_graph.nodes) < k_mid):
            k_max = k_mid
        else: 
            k_min = k_mid
    return k_min

def k_d_reduce_graph(clique_graph, k, d):
    nodes_before = len(clique_graph.nodes)+1
    nodes_after = len(clique_graph.nodes)
    false_sets = set()
    while(nodes_before > nodes_after or edges_before > edges_after):
        nodes_before = nodes_after
        k_core_clique_graph = nx.k_core(clique_graph, k-1)
        nodes_after = len(k_core_clique_graph.nodes)

        edges_before = len(k_core_clique_graph.edges)
        k_edge_core_clique_graph, false_sets = k_d_edge_core(k_core_clique_graph, k=k-2, d=d, false_sets=false_sets)
        #k_edge_core_clique_graph = k_d_edge_core(k_core_clique_graph, k=k-2, d=d)
        edges_after = len(k_edge_core_clique_graph.edges)

        clique_graph = k_core_clique_graph
    return clique_graph

def k_d_edge_core(clique_graph, k, d, node_tuple = (), previous_common_neighbors = [], false_sets = set()):
    if(len(node_tuple) == 0):
        edges_before = len(clique_graph.edges)+1
        edges_after = len(clique_graph.edges)
        while(edges_before > edges_after):
            edges_before = edges_after
            for edge in tqdm(clique_graph.edges, disable=True):
                common_neighbors = set(nx.common_neighbors(clique_graph, *edge))
                if(len(common_neighbors) < k):
                    clique_graph.remove_edge(*edge)
                else:
                    condition = k_d_edge_core(clique_graph, k-1, d=d, node_tuple=tuple(sorted((*edge,))), previous_common_neighbors=common_neighbors, false_sets=false_sets)
                    if(not condition):
                        clique_graph.remove_edge(*edge)
            edges_after = len(clique_graph.edges)
        return clique_graph, false_sets
    
    elif(len(node_tuple)-1 <= d):
        for node in previous_common_neighbors:
            common_neighbors = previous_common_neighbors.intersection(set(clique_graph.neighbors(node)))
            if(len(common_neighbors) >= k):
                new_nodes_tuple = tuple(sorted(node_tuple+(node,)))
                if(new_nodes_tuple not in false_sets):
                    condition = k_d_edge_core(clique_graph, k-1, d=d, node_tuple=new_nodes_tuple, previous_common_neighbors=common_neighbors, false_sets=false_sets)
                    if(condition):
                        return True
                    else:
                        false_sets.add(new_nodes_tuple)
        return False
    return True

def __k_d_edge_core(clique_graph, k, d, node_tuple = (), previous_common_neighbors = [], false_sets = set()):
    if(len(node_tuple) == 0):
        edges_before = len(clique_graph.edges)+1
        edges_after = len(clique_graph.edges)
        while(edges_before > edges_after):
            edges_before = edges_after
            for edge in tqdm(clique_graph.edges, disable=True):
                common_neighbors = set(nx.common_neighbors(clique_graph, *edge))
                if(len(common_neighbors) < k):
                    clique_graph.remove_edge(*edge)
                else:
                    condition = k_d_edge_core(clique_graph, k-1, d=d, node_tuple=tuple(sorted((*edge,))), previous_common_neighbors=common_neighbors, false_sets=false_sets)
                    if(not condition):
                        clique_graph.remove_edge(*edge)
            edges_after = len(clique_graph.edges)
        #return clique_graph, false_sets
        return clique_graph
    
    elif(len(node_tuple)-1 <= d):
        for node in previous_common_neighbors:
            common_neighbors = previous_common_neighbors.intersection(set(clique_graph.neighbors(node)))
            if(len(common_neighbors) >= k):
                new_nodes_tuple = tuple(sorted(node_tuple+(node,)))
                if(new_nodes_tuple not in false_sets):
                    condition = k_d_edge_core(clique_graph, k-1, d=d, node_tuple=new_nodes_tuple, previous_common_neighbors=common_neighbors, false_sets=false_sets)
                    if(condition):
                        return True
                    else:
                        false_sets.add(new_nodes_tuple)
        return False
    return True