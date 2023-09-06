# Table of contents
- [Introduction](#introduction)
- [Installing requirements](#installing-requirements)
- [Stable set solver library](#stable-set-solver-library)
  - [solution_correction](#solution_correction)
  - [calculate_partition_with_halo](#calculate_partition_with_halo)
  - [print_best_partitions](#print_best_partitions)
  - [calculate_best_solution](#calculate_best_solution)
  - [open_saved_sample_set](#open_saved_sample_set)
  - [eliminate_and_recalculate](#eliminate_and_recalculate)
- [Examples](#examples)

# Introduction
This library was written as the supplementary material for the bachelor thesis: "Reševanje NP-težkih optimizacijskih problemov z uporabo kvantnih žarilnikov" or, in english "Solving NP-hard optimization problems using quantum annealers". As the name suggests, it focuses on solving optimization problems on (D-Wave's) quantum annealers, in our case, the main focus is on developing theory for and solving the maximum stable set problem (also known as the maximum independet set). But the general problem solving techniques and ideas used can also be translated to any other problem. This library provides the code that was used to calcualte the result, the code reusable and can be used for an arbitrary (simple, undirected) graph.

# Installing requirements
You can install requirements with the following code:
```
pip install requirements.txt
```
Note: `metis` library might not work on windows, if you don't manage to set it up, use linux, or just remove the `from metis import part_graph` from the code, but this will mean that you won't be able to use features that make use of multiple partitions.

# Stable set solver library
For using the library, put the stable_set_solver.py in the same folder as your main file, and then import it using:
```
import stable_set_solver as sss
```
The following functions are available:

## solution_correction
```
def solution_correction(
        stable_set_graph,
        solution_nodes
    ):
```
Argument | Explanation
---|---
`stable_set_graph`| `networkx` graph, on which `solution_nodes` were calculated
`solution_nodes`| list of nodes, that are in solution

Returns: `corrected_solution`, which is a list of nodes, that are a subset of `solution_nodes` and are stable set in graph `stable_set_graph`. Properties of this algorithm are described in proposition 4.1, and how its solution relates to energy in `stable_set_graph` is described in proposition 4.2.

## calculate_partition_with_halo
```
def _calculate_partition_with_halo(
        stable_set_graph,
        num_of_part
    ):
```
Argument | Explanation
---|---
`stable_set_graph`| `networkx` graph, on which partition will be calculated
`num_of_part`| number of partitions wanted

Return: List of set of nodes, one for each partition.
## print_best_partitions
```
def print_best_partitions(
        stable_set_graph,
        max_num_of_part,
        sort='v'
    ):
```
Argument | Explanation
---|---
`stable_set_graph`| `networkx` graph, on which partitions will be made
`max_num_of_part`| Function will calculate partitions in range `2, ..., max_num_of_part`
`sort='v'`| Sort the partitions in descending order (so that best solutions appear on the bottom of the terminal) by max vertex number with `sort='v'`, or by max edge number with `sort='e'`. For any other input nothing happens.

## calculate_best_solution
```
def calculate_best_solution(
        stable_set_graph,
        sampler,
        num_of_runs=1,
        num_of_part=1,
        output_file=None,
        no_output_file = False,
        console_output=True
    ):
```

Argument | Explanation
---|---
`stable_set_graph`| `networkx` graph, on which stable set will be calculated
`sampler`| D-Wave's sampler for solving. For the list refer to their [documentation](https://docs.dwavesys.com/docs/latest/c_gs_workflow.html#samplers)
`num_of_runs=1`| Number of runs that sampler does, note da some sampler (for example, hybrid) does not support this argument, so in that case you have to leave it on 1, otherwise an error will occur.
`num_of_part=1`| Number of partitions graph will be partitioned.
`output_file=None`| Path to the name of the output file. Note that if no path is given, directory does not exist, an exception will occur. This is to prevent the accidental lost of result when computing. Output file can be omitted by setting `no_output_file=True`.
`no_output_file=False`| If set to `True`, no output file will be generated
`console_output=True`| This controls whether console outputs the results or not.

Returns: a list `all_solutions_info` where each entry corresponds to the partition and contains solution information dictionary. Each dictionary contains the following keys:

Key | Explanation
---|---
`"best_solution_nodes"`| a list of nodes in the best solution
`"best_solution_energy"`| energy value of the best solution
`"best_uncorrected_solution_nodes"`| a list of nodes in the best solution before correction
`"best_uncorrected_solution_energy"`| energy value of the best solution before correction
`"sample_set"`| `SampleSet` object for the partition

`best_uncorrected_solution` is the one that we get directly from the sampler, and it might contain some edges, meaning it's not a stable set. We get the `best_solution` by submitting `best_uncorrected_solution` to the `solution_correction` function, which properties are described in the proposition 4.1 of the thesis.

## open_saved_sample_set
```
def open_saved_sample_set(input_file):
```
Argument | Explanation
---|---
`input_file`| a path to .pkl file that was produced by `calculate_best_solution` function.

Returns: `SampleSet` object.

## eliminate_and_recalculate

```
def eliminate_and_recalculate(
        stable_set_graph,
        sample_set,
        sampler=None,
        num_of_runs=1
    ):
```

Argument | Explanation
---|---
`stable_set_graph`| `networkx` graph, on which `sample_set` was calculated on.
`sample_set`|`SampleSet` object on which to eliminate samples and recalculate ones who we couldn't eliminate
`sampler=None`| D-Wave's sampler for recalculations. For the list refer to their [documentation](https://docs.dwavesys.com/docs/latest/c_gs_workflow.html#samplers). If left empty, it will default to the `solution_correction` function algorithm for getting the stable set.
`num_of_runs=1`| Number of runs that sampler does, note da some sampler (for example, hybrid) does not support this argument, so in that case you have to leave it on 1, otherwise an error will occur. This has no effect if no sampler is provided.

Returns a dictionary with keys:
Key | Explanation
---|---
`"best_recalculated_solution"`| dictionary `recalculated_solutions` = {`"recalculated_solution_nodes"`: `list_of_nodes`, `"recalculated_solution_energy"`: `solution_energy`}
`"all_recalculated_solutions"`| list of dictionaries `recalculated_solutions`, define like in above key

# Examples
Examples are located in the `examples.py` file:
 - `example1` shows the use of `calculate_best_solution` function without partitionig (`num_of_part=1`) for hybrid and QPU sampler.
 - `example2` shows the use of `eliminate_and_recalculate` function for the results we got from QPU sampler in `example1`.
 - `example3` shows the use of `print_best_partitions` function on 2 dense graphs.
 - `example4` shows the use of `calculate_best_solution` function with partitioning, we use suitable partition sizes we got from `example3`