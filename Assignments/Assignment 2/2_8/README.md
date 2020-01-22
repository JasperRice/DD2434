## DD2434 Assignment 2 
### Task 2.8

For this task you are supposed to implement algorithms for dynamic programming and sampling.
In addition, you should sample n (at least 1000) sets of leaves from the provided tree, and compute the proportion
of leaf sets that fall into the provided interval (list of values for each node).

#### Loading trees

The file 2_8.py contains help functions for creating trees or loading the provided tree for final results.
You can play around with different tree structures using
```
binary_tree = test_binary_trees(k, num_nodes)
```
The tree that should be used for presenting results can be loaded with
```
binary_tree = load_tree()
```
The tree is a Tree object from Tree.py, and you can investigate the file for more details on the class.
Note that you can choose not to use the help functions or the tree class. The file for the tree that
you should use to present results is called 'tree_task_2_8' and comes in pickle and newick string format.
The algorithm implementations can be written in 2_8.py or a separate file.

#### Loading interval dictionaries and testing algorithm

Once your implementation is complete, there are two dictionaries that can be loaded with the helper function
```
most_freq_dict = load_interval('interval_task_2_8.npy')
```
Use your sampling function to sample n sets of leaves and compute the following values:

For the dictionary 'interval_task_2_8' compute the proportion of sampled leaf sets that matches the values of the dictionary.
The dictionary contains keys for each leaf node, and each node has a list of values. If each of your sampled leaf values of a single
set can be found in the dictionary, that sampled set is within the interval. E.g.

```
sampled_leaves = {'5': 3, '4': 2}
dictionary = {'5': [2, 3], '4': [1, 3, 5]}
```
would mean that that sample didn't fully fall into the value sets of the dictionary.

For 'node_interval_task_2_8' we compute the same but for each leaf node separately instead of the whole set. In other words, for each
leaf node we wish to compute the proportion of sampled values that are found in the dictionary. The result should be a value for each leaf node. E.g.
```
sampled_leaves = {'5': 3, '4': 2}
dictionary = {'5': [2, 3], '4': [1, 3, 5]}
```
means node '5' falls into the dictionary, but '4' doesn't.
The file 2_8.py contains functions to give you the idea, although it expects your samples to be as a dictionary with node names as keys.
