"""
File for task 2.8 in assignment 2 of DD2434.
See README for more instructions and information on generating/loading trees.
"""

import numpy as np
from Tree import Tree, Node

def test_binary_trees(seed_val=0, k=2, num_nodes=5):
    """
    Test function for generating binary trees from Tree.py and printing topology.
    :param: seed_val: Numpy seed.
    :param: k: Number of possible values for a node.
    :param: num_nodes: Number of total nodes in the tree. Has to be odd, or it will auto correct to +1.
    :return: binary_tree: A generated binary tree. Tree object from Tree.py.
    """
    binary_tree = Tree()
    binary_tree.create_random_binary_tree(seed_val=seed_val, k=k, num_nodes=num_nodes)

    # Print tree
    binary_tree.print()

    # Print topology
    binary_tree.print_topology()

    return binary_tree

def load_tree():
    """
    Function for loading the tree for the assignment.
    :return: Loaded tree
    """
    binary_tree = Tree()
    binary_tree.load_tree('tree_task_2_8')

    # Print tree
    binary_tree.print()

    # Print topology
    binary_tree.print_topology()

    return binary_tree

def load_interval(path='interval_task_2_8.npy', str=False):
    """
    Function for loading the interval dictionary for the assignment.
    :param: str: Load from text file or numpy file.
    :return: Dictionary of interval.
    """
    if not str:
        mode_dict = np.load(path, allow_pickle=True).item()
    else:
        from ast import literal_eval
        with open(path, "r") as file:
            data = file.readlines()
        mode_dict = literal_eval(data[0])
    return mode_dict

def test_node_sample_proportion(dict, tree, n=1000):
    """
    Function for sampling n leaf sets and computing, for each leaf node,
    the proportion of samples that are in the provided interval.
    :param: dict: Interval dictionary containing node keys and lists of possible node values.
    :param: tree: Generated tree. Tree object from Tree.py.
    :param: n: Number of samples.
    :return: Dictionary with proportion of samples that are in the interval for each node.
    """
    n_samples = {}
    for i in range(n):
        # leaf_samples = <SAMPLING FUNCTION HERE>
        if i == 0:
            n_samples = {key: 0 for key in leaf_samples.keys()}
        for key, val in leaf_samples.items():
            if val in dict[key]:
                n_samples[key] += 1/n
    return n_samples

def test_sample_proportion(dict, tree, n=1000):
    """
    Function for sampling n leaf sets and computing the proportion of sets that are in the interval dict.
    :param: dict: Interval dictionary containing node keys and lists of possible node values.
    :param: tree: Generated tree. Tree object from Tree.py.
    :param: n: Number of samples.
    :return: Proportion of samples that are in interval dict.
    """
    n_samples = 0
    for i in range(n):
        # leaf_samples = <SAMPLING FUNCTION HERE>
        for key, val in leaf_samples.items():
            if val not in dict[key]:
                break
        else:
            n_samples += 1
    return n_samples/n

def tree_DP(root, k):
    # TODO: Implement algorithm for dynamic programming
    p = np.dot(root.cat, odd_DP(root, k))
    q = np.dot(root.cat, even_DP(root, k))
    print(p+q)
    return p

def odd_DP(node, k):
    if len(node.descendants) != 0:
        theta1 = np.array((node.descendants[0].cat))
        theta2 = np.array((node.descendants[1].cat))

        s11 = odd_DP(node.descendants[0], k)
        s21 = even_DP(node.descendants[1], k)
        s1  = np.dot(theta1, s11) * np.dot(theta2, s21)
        
        s12 = even_DP(node.descendants[0], k)
        s22 = odd_DP(node.descendants[1], k)
        s2  = np.dot(theta1, s12) * np.dot(theta2, s22)
        return s1 + s2

    s = np.zeros((k,1))
    s[1::2] = 1
    return s

def even_DP(node, k):
    if len(node.descendants) != 0:
        theta1 = np.array((node.descendants[0].cat))
        theta2 = np.array((node.descendants[1].cat))

        s11 = odd_DP(node.descendants[0], k) 
        s21 = odd_DP(node.descendants[1], k)
        s1  = np.dot(theta1, s11) * np.dot(theta2, s21)
        
        s12 = even_DP(node.descendants[0], k)
        s22 = even_DP(node.descendants[1], k)
        s2  = np.dot(theta1, s12) * np.dot(theta2, s22)
        return s1 + s2

    s = np.zeros((k,1))
    s[0::2] = 1
    return s

def odd_sum_sampling(root, K):
    # TODO: Implement sampling algorithm
    

    sample = np.zeros(K)

def main():
    # Test tree
    k = 5
    num_nodes = 100
    binary_tree = test_binary_trees(seed_val=0, k=k, num_nodes=num_nodes)
    p = tree_DP(binary_tree.root, k)
    print(p)


    # Load tree
    # binary_tree = load_tree()

    # Load interval dictionary
    # most_freq_dict = load_interval('interval_task_2_8.npy')
    # node_most_freq_dict = load_interval('node_interval_task_2_8.npy')


if __name__ == "__main__":
    main()