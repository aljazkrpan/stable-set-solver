import networkx as nx
from pathlib import Path
from tqdm import tqdm

from stable_set_solver import calculate_best_solution
from stable_set_solver import open_saved_sample_set
from stable_set_solver import eliminate_and_recalculate
from stable_set_solver import print_best_partitions
from stable_set_solver import k_edge_core
from stable_set_solver import k_reduce_graph
from stable_set_solver import first_k_for_reduction
from stable_set_solver import k_reduce_upper_bound
from stable_set_solver import k_d_reduce_upper_bound
from stable_set_solver import k_d_reduce_graph

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import AutoEmbeddingComposite
from dwave.system import LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler

all_graph_files = [
    'Stable_set_data-main/Instances/C_graphs/C125.9_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/dsjc_graphs/dsjc125.1_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/dsjc_graphs/dsjc125.5_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/hamming_graphs/hamming6_2_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/hamming_graphs/hamming6_4_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/johnson_graphs/johnson8_2_4_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/johnson_graphs/johnson8_4_4_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/johnson_graphs/johnson16_2_4_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/MANN_graphs/MANN_a9_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley61_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley73_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley89_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley97_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley101_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/spin_graphs/spin5_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/torus_graphs/torus11_stable_set_edge_list.txt'
]

all_graph_files_with_best_sol = [
    ('Stable_set_data-main/Instances/C_graphs/C125.9_stable_set_edge_list.txt', 34),
    ('Stable_set_data-main/Instances/dsjc_graphs/dsjc125.1_stable_set_edge_list.txt', 34),
    ('Stable_set_data-main/Instances/dsjc_graphs/dsjc125.5_stable_set_edge_list.txt', 10),
    ('Stable_set_data-main/Instances/hamming_graphs/hamming6_2_stable_set_edge_list.txt', 32),
    ('Stable_set_data-main/Instances/hamming_graphs/hamming6_4_stable_set_edge_list.txt', 4),
    ('Stable_set_data-main/Instances/johnson_graphs/johnson8_2_4_stable_set_edge_list.txt', 4),
    ('Stable_set_data-main/Instances/johnson_graphs/johnson8_4_4_stable_set_edge_list.txt', 14),
    ('Stable_set_data-main/Instances/johnson_graphs/johnson16_2_4_stable_set_edge_list.txt', 8),
    ('Stable_set_data-main/Instances/MANN_graphs/MANN_a9_stable_set_edge_list.txt', 16),
    ('Stable_set_data-main/Instances/paley_graphs/paley61_stable_set_edge_list.txt', 5),
    ('Stable_set_data-main/Instances/paley_graphs/paley73_stable_set_edge_list.txt', 5),
    ('Stable_set_data-main/Instances/paley_graphs/paley89_stable_set_edge_list.txt', 5),
    ('Stable_set_data-main/Instances/paley_graphs/paley97_stable_set_edge_list.txt', 6),
    ('Stable_set_data-main/Instances/paley_graphs/paley101_stable_set_edge_list.txt', 5),
    ('Stable_set_data-main/Instances/spin_graphs/spin5_stable_set_edge_list.txt', 50),
    ('Stable_set_data-main/Instances/torus_graphs/torus11_stable_set_edge_list.txt', 55)
]

all_dense_graph_files = [
    'Stable_set_data-main/Instances/c_fat_graphs/c_fat200_2_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/c_fat_graphs/c_fat500_5_stable_set_edge_list.txt'
]

all_dense_graph_files_with_best_sol = [
    ('Stable_set_data-main/Instances/c_fat_graphs/c_fat200_2_stable_set_edge_list.txt', 24),
    ('Stable_set_data-main/Instances/c_fat_graphs/c_fat500_5_stable_set_edge_list.txt', 63)
]

# Reads edge list from the Stable_set_data-main dataset
def read_edge_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_nodes, num_edges = map(int, lines[0].split())

    edges = []
    for line in lines[1:]:
        u, v, w = map(int, line.split())
        edges.append((u-1, v-1))
    
    return num_nodes, edges

# Creates graph from the edges we read
def create_networkx_graph(num_nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(0, num_nodes))
    G.add_edges_from(edges)
    return G

# example1 just solves the problem directly and outputs the solution with the best energy. Since we are using
# the platform time, we create a file for each response we get, just in case we might need something else later on
def example1():
    global all_graph_files
    for edge_list_file_path in all_graph_files:
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)

        # Here we calculate results on the QPU solver which get saved in the file
        print(f"Using QPU solver on {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')}")
        # Or if you don't have an account you can replace it with: sampler = SimulatedAnnealingSampler()
        sampler = AutoEmbeddingComposite(DWaveSampler(token="provide_you_token_here"))
        num_of_runs = 1000
        output = f"results/example_results/QPU_1/{Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')}_runs{num_of_runs}.pkl"
        solutions = calculate_best_solution(stable_set_graph, sampler, output_file=output, num_of_runs=num_of_runs)

        # Here we calculate results on the hybrid solver which get saved in the file
        print(f"Using hybrid solver on {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')}")
        # or if you don't have an account you can replace it with: sampler = SimulatedAnnealingSampler()
        sampler = LeapHybridSampler(token="provide_you_token_here")
        output = fr"results/example_results/hybrid_1/{Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')}.pkl"
        solutions = calculate_best_solution(stable_set_graph, sampler, output_file=output)

# example2 uses solutions that we saved in the example1. It goes through all the solutions and picks only the
# potential ones that might improve our result, and then it does recalculation based on these solutions. More detail
# on how and why this works is described in the document in section 4.3
def example2():
    global all_graph_files
    for edge_list_file_path in all_graph_files:
        response = open_saved_sample_set(f"results/example_results/QPU_1/{Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')}_runs1000.pkl")
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)

        # if we leave the sampler empty, it will default to the algorithm described in the proposition 4.1
        rec_sol = eliminate_and_recalculate(stable_set_graph, response)
        print(f"In graph {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')} || calculated on {len(rec_sol['all_recalculated_solutions'])} samples || best recalculated solution: {rec_sol['best_recalculated_solution']['recalcualted_solution_energy']} using default algorithm")

        # Otherwise it will just use whatever we give it
        rec_sol = eliminate_and_recalculate(stable_set_graph, response, sampler=SimulatedAnnealingSampler(), num_of_runs=10)
        print(f"In graph {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')} || calculated on {len(rec_sol['all_recalculated_solutions'])} samples || best recalculated solution: {rec_sol['best_recalculated_solution']['recalcualted_solution_energy']} || using simulated annealing")

# example3 just outputs the the CH-partition information for each partition size between 2, ... , max_num_of_part that we get
# from the METIS algorithm. This works best on dense graphs (those with a lot of edges)
def example3():
    # for both graph we print out the best partitions sorted by cost, so that we can decide which one to use
    global all_dense_graph_files
    for edge_list_file_path in all_dense_graph_files:
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)
        print_best_partitions(stable_set_graph, max_num_of_part=50)

# example4 uses suitable partition from the example3 to try to improve the solution.
def example4():
    # or if you don't have an account you can replace it with:
    # sampler = SimulatedAnnealingSampler()
    sampler = AutoEmbeddingComposite(DWaveSampler(token="provide_you_token_here"))

    # for fat200_2, we'll use partition size 9 we got from example3()
    edge_list_file_path = 'Stable_set_data-main/Instances/c_fat_graphs/c_fat200_2_stable_set_edge_list.txt'
    num_nodes, edges = read_edge_list(edge_list_file_path)
    stable_set_graph = create_networkx_graph(num_nodes, edges)
    output = "results/example_results/simplified_with_partitions/QPU/c_fat200_2_runs1000/c_fat200_2.pkl"
    solution = calculate_best_solution(stable_set_graph, sampler, output_file=output, num_of_part=9, num_of_runs=1000)

    # for fat500_5, we'll use partition size 8 we got from example3()
    edge_list_file_path = 'Stable_set_data-main/Instances/c_fat_graphs/c_fat500_5_stable_set_edge_list.txt'
    num_nodes, edges = read_edge_list(edge_list_file_path)
    stable_set_graph = create_networkx_graph(num_nodes, edges)
    output = "results/example_results/simplified_with_partitions/QPU/c_fat500_5_runs1000/c_fat500_5.pkl"
    solution = calculate_best_solution(stable_set_graph, sampler, output_file=output, num_of_part=8, num_of_runs=1000)

# We use k_reduction to reduce graph with k=64, and then use suitable partition to on the reduced graph
def example5():
    edge_list_file_path = 'Stable_set_data-main/Instances/c_fat_graphs/c_fat500_5_stable_set_edge_list.txt'
    num_nodes, edges = read_edge_list(edge_list_file_path)
    stable_set_graph = create_networkx_graph(num_nodes, edges)
    clique_graph = nx.complement(stable_set_graph)
    reduced_clique_graph = k_reduce_graph(clique_graph, 64)
    reduced_stable_set_graph = nx.complement(reduced_clique_graph)
    print(f"Reduced stable set graph nodes {len(reduced_stable_set_graph.nodes)} and edges {len(reduced_stable_set_graph.edges)}")
    print_best_partitions(reduced_stable_set_graph, max_num_of_part=50)
    num_of_part = int(input("Enter the number of partitions: "))
    
    sampler = AutoEmbeddingComposite(DWaveSampler(token="provide_you_token_here"))
    output = "results/example_results/k_reduced_and_simplified_with_partitions/QPU/c_fat500_5_runs1000/c_fat500_5"
    solution = calculate_best_solution(reduced_stable_set_graph, sampler, output_file=output, num_of_part=num_of_part, num_of_runs=1000)

# We use k_core_upper_bound to find upper bound for stable set problem using the method described in the thesis
def example6():
    for edge_list_file_path in all_dense_graph_files + all_graph_files:
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)
        
        edge_ratio =  round(len(stable_set_graph.edges)/(len(stable_set_graph.nodes)*(len(stable_set_graph.nodes)-1)*0.5), 2)
        print(f"Upper bound for graph {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')} with edge ratio {edge_ratio} is {k_reduce_upper_bound(stable_set_graph)}")

# We use k_core_upper_bound together with CH-partitions to find upper bound for stable set problem using the method described in the thesis
def example7():
    for edge_list_file_path in all_graph_files[1:]:
        print(f"Started calculating upper bound for graph {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')}")
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)
        clique_graph = nx.complement(stable_set_graph)
        partition_with_halo_list = list(map(lambda x : list(set(clique_graph.neighbors(x)).union({x})), clique_graph.nodes))
        #upper_bounds_of_subgraphs = map(lambda x : find_k_core_upper_bound(stable_set_graph.subgraph(x)), partition_with_halo_list)
        upper_bounds_of_subgraphs = []
        for partition in tqdm(partition_with_halo_list):
            upper_bounds_of_subgraphs.append(k_reduce_upper_bound(stable_set_graph.subgraph(partition)))
        
        edge_ratio =  round(len(stable_set_graph.edges)/(len(stable_set_graph.nodes)*(len(stable_set_graph.nodes)-1)*0.5), 2)
        print(f"Upper bound with k-reductions and partitions for graph {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')} with edge ratio {edge_ratio} is {max(upper_bounds_of_subgraphs)}")

# Using generalized version of k_d_reduce we find upper bound for d=2, just like described in the thesis
def example8():        
    for edge_list_file_path in all_graph_files[8:9]:
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)
        clique_graph = nx.complement(stable_set_graph)
        partition_with_halo_list = list(map(lambda x : list(set(clique_graph.neighbors(x)).union({x})), clique_graph.nodes))
        upper_bounds_of_subgraphs = []
        depth = 2
        print(f"Upper bound with k,{depth}-reductions and partitions for graph {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')}")
        for partition in tqdm(partition_with_halo_list):
            upper_bounds_of_subgraphs.append(k_d_reduce_upper_bound(stable_set_graph.subgraph(partition), d=depth))
        
        edge_ratio =  round(len(stable_set_graph.edges)/(len(stable_set_graph.nodes)*(len(stable_set_graph.nodes)-1)*0.5), 2)
        print(f"Upper bound with k,{depth}-reductions and partitions for graph {Path(edge_list_file_path).name.replace('_stable_set_edge_list.txt','')} with edge ratio {edge_ratio} is {max(upper_bounds_of_subgraphs)}")

if __name__ == "__main__":
    example1()