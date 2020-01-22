""" This file created as supplementary code for tree-related questions in DD2434 - Assignment 2.
    This file demonstrates how to use the DendroPy module to load the trees in Newick format
    and calculate the Robinson-Foulds (RF) distance.

    More information can be found:
    https://en.wikipedia.org/wiki/Robinson%E2%80%93Foulds_metric
    https://dendropy.org/library/treecompare.html

    There are other phylogeny modules in Python and R.
    You can use any of the modules and if you want, you can implement your version of RF calculation as well. """

import dendropy
import numpy as np
from Tree import Tree


def tree_to_newick_rec(cur_node):
    items = []
    num_children = len(cur_node.descendants)
    for child_idx in range(num_children):
        s = ''
        sub_tree = tree_to_newick_rec(cur_node.descendants[child_idx])
        if sub_tree != '':
            s += '(' + sub_tree + ')'
        s += cur_node.descendants[child_idx].name
        items.append(s)
    return ','.join(items)


def main():
    print("Hello World!")
    print("This file demonstrates the usage of the DendroPy module and its functions.")

    print("\n1. Tree Generations\n")
    print("\n1.1. Create two random birth-death trees and print them:\n")

    # If you want to compare two trees, make sure you specify the same Taxon Namespace!
    tns = dendropy.TaxonNamespace()
    num_leaves = 5

    t1 = dendropy.simulate.treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=num_leaves,
                                                    taxon_namespace=tns)
    t2 = dendropy.simulate.treesim.birth_death_tree(birth_rate=1.0, death_rate=0.2, num_extant_tips=num_leaves,
                                                    taxon_namespace=tns)
    print("\tTree 1: ", t1.as_string("newick"))
    t1.print_plot()
    print("\tTree 2: ", t2.as_string("newick"))
    t2.print_plot()

    print("\n2. Compare Trees\n")
    print("\n2.1. Compare tree with itself and print Robinson-Foulds (RF) distance:\n")

    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, t1))

    print("\n2.2. Compare different trees and print Robinson-Foulds (RF) distance:\n")

    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, t2))

    print("\n3. Load Trees from Newick Files and Compare:\n")
    print("\n3.1 Load trees from Newick files:\n")

    # If you want to compare two trees, make sure you specify the same Taxon Namespace!
    tns = dendropy.TaxonNamespace()

    filename = "data/example_tree_mixture.pkl_tree_0_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t0 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 0: ", t0.as_string("newick"))
    t0.print_plot()

    filename = "data/example_tree_mixture.pkl_tree_1_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t1 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 1: ", t1.as_string("newick"))
    t1.print_plot()

    filename = "data/example_tree_mixture.pkl_tree_2_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t2 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 2: ", t2.as_string("newick"))
    t2.print_plot()

    print("\n3.2 Compare trees and print Robinson-Foulds (RF) distance:\n")

    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, t1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, t2))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, t2))

    print("\n4. Load Inferred Trees")
    filename = "data/example_result_em_topology.npy"
    topology_list = np.load(filename)
    print(topology_list.shape)
    print(topology_list)

    rt0 = Tree()
    rt0.load_tree_from_direct_arrays(topology_list[0])
    rt0 = dendropy.Tree.get(data=rt0.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred Tree 0: ", rt0.as_string("newick"))
    rt0.print_plot()

    rt1 = Tree()
    rt1.load_tree_from_direct_arrays(topology_list[1])
    rt1 = dendropy.Tree.get(data=rt1.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred Tree 1: ", rt1.as_string("newick"))
    rt1.print_plot()

    rt2 = Tree()
    rt2.load_tree_from_direct_arrays(topology_list[2])
    rt2 = dendropy.Tree.get(data=rt2.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred Tree 2: ", rt2.as_string("newick"))
    rt2.print_plot()

    print("\n4.2 Compare trees and print Robinson-Foulds (RF) distance:\n")

    print("\tt0 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt2))

    print("\tt1 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt2))

    print("\tt2 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt2))

    print("\nInvestigate")

    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.find_missing_bipartitions(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.false_positives_and_negatives(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.find_missing_bipartitions(t0, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.false_positives_and_negatives(t0, rt1))

if __name__ == "__main__":
    main()
