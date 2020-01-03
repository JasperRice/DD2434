""" This file is created as a template for question 2.5 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_5.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def em_algorithm(seed_val, samples, k, num_iter=10):
    You can change it to: def em_algorithm(seed_val, samples, k, num_iter=10, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)!

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    After all, we will test your code with commands like this one:
    %run 2_5.py "data/example_tree_mixture.pkl_samples.txt" "data/example_result" 3 --seed_val 123
    where
    "data/example_tree_mixture.pkl_samples.txt" is the filename of the samples
    "data/example_result" is the base filename of results (i.e data/example_result_em_loglikelihood.npy)
    3 is the number of clusters for EM algorithm
    --seed_val is the seed value for your code, for reproducibility.

    For this assignment, we gave you three different trees
    (q_2_5_tm_10node_20sample_4clusters, q_2_5_tm_10node_50sample_4clusters, q_2_5_tm_20node_20sample_4clusters).
    As the names indicate, the mixtures have 4 clusters with varying number of nodes and samples.
    We want you to run your EM algorithm and compare the real and inferred results in terms of Robinson-Foulds metric
    and the likelihoods.
    """
import argparse
import dendropy
import numpy as np
import matplotlib.pyplot as plt

from Tree import TreeMixture, Tree, Node

def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)


def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    You can change the function signature and add new parameters. Add them as parameters with some default values.
    i.e.
    Function template: def em_algorithm(seed_val, samples, k, max_num_iter=10):
    You can change it to: def em_algorithm(seed_val, samples, k, max_num_iter=10, new_param_1=[], new_param_2=123):
    """
    from Tree import TreeMixture, Tree
    from Kruskal_v1 import Graph
    import sys
    np.random.seed(seed_val)
    epsilon = sys.float_info.epsilon
    # epsilon = sys.float_info.min

    print("Running EM algorithm...")

    # Set threshold for convergence
    THRES = 1e-4

    # Get num_samples and num_nodes from samples
    num_samples = np.size(samples, 0)
    num_nodes = np.size(samples, 1)

    # Initialize trees
    tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
    tm.simulate_pi(seed_val=seed_val)
    tm.simulate_trees(seed_val=seed_val)
    # tm.sample_mixtures(num_samples=num_samples, seed_val=seed_val)

    loglikelihood = []
    for iter in range(max_num_iter):
        print("==================== "+str(iter)+" ====================")
        # Step 1: Compute the responsibilities
        print("=> Computing responsibilities...")
        r = np.ones((num_samples, num_clusters))
        for n, x in enumerate(samples):
            for k, t in enumerate(tm.clusters):
                r[n,k] *= tm.pi[k]
                visit_list = [t.root]
                while len(visit_list) != 0:
                    cur_node = visit_list[0]
                    visit_list = visit_list[1:]
                    visit_list = visit_list + cur_node.descendants
                    if cur_node.ancestor is None:
                        r[n,k] *= cur_node.cat[x[int(cur_node.name)]]
                    else:
                        r[n,k] *= cur_node.cat[x[int(cur_node.ancestor.name)]][x[int(cur_node.name)]]
        r += epsilon
        
        marginal = np.reshape(np.sum(r, axis=1), (num_samples,1))
        loglikelihood.append(np.sum(np.log(marginal)))
        # TODO: Judge whether converge:
        
        marginal = np.repeat(marginal, num_clusters, axis=1)
        r /= marginal

        # Step 2: Update categorical distribution
        print("=> Updating categorical distribution...")
        tm.pi = np.mean(r, axis=0)

        # Step 3: Construct directed graphs
        print("=> Constructing directed graphs...")
        denom = np.sum(r, axis=0)
        q = np.zeros((num_nodes, num_nodes, 2, 2, num_clusters)) # (s, t, a, b, k)
        for s in range(num_nodes):
            for t in range(num_nodes):
                for a in range(2):
                    for b in range(2):
                        index = np.where((samples[:,(s,t)]==[a,b]).all(1))[0]
                        numer = np.sum(r[index], axis=0)
                        q[s, t, a, b] = numer / denom
        q += epsilon
        
        q_s = np.zeros((num_nodes, 2, num_clusters))
        for s in range(num_nodes):
            for a in range(2):
                index = np.where(samples[:, s]==a)
                numer = np.sum(r[index], axis=0)
                q_s[s,a] = numer / denom
        q_s += epsilon

        I = np.zeros((num_nodes, num_nodes, num_clusters)) # (s, t, k)
        for s in range(num_nodes):
            for t in range(num_nodes):
                for a in range(2):
                    for b in range(2):
                        I[s,t] += q[s,t,a,b] * np.log(q[s,t,a,b] / q_s[s,a] / q_s[t,b])

        clusters = []
        for k in range(num_clusters):
            g = Graph(num_nodes)
            for s in range(num_nodes):
                for t in range(s+1, num_nodes):
                    g.addEdge(s, t, I[s, t, k])

            # Step 4: Construct maximum spanning trees
            print("=> Constructing maximum spanning trees...")
            edges = np.array(g.maximum_spanning_tree())[:,0:2]
            # topology_array = num_nodes * np.ones(num_nodes)
            topology_array = np.zeros(num_nodes)
            topology_array[0] = np.nan
            visit_list = [0]
            while len(visit_list) != 0:
                cur_node = visit_list[0]
                index = np.where(edges==cur_node)
                index = np.transpose(np.stack(index))
                visit_list = visit_list[1:]
                for id in index:
                    child = edges[id[0], 1-id[1]]
                    topology_array[int(child)] = cur_node
                    visit_list.append(int(child))
                if np.size(index) is not 0:
                    edges = np.delete(edges, index[:,0], axis=0)

            tree = Tree()
            tree.load_tree_from_direct_arrays(topology_array)
            tree.k = 2
            tree.alpha = [1.0] * 2

            # Step 5: Update CPDs
            print("=> Updating CPDs...")
            visit_list = [tree.root]
            while len(visit_list) != 0:
                cur_node = visit_list[0]
                visit_list = visit_list[1:]
                visit_list = visit_list + cur_node.descendants
                if cur_node.ancestor is None:
                    cur_node.cat = q_s[int(cur_node.name),:,k].tolist()
                else:
                    cat = q[int(cur_node.ancestor.name),int(cur_node.name),:,:,k]
                    cur_node.cat = [cat[0].tolist(), cat[1].tolist()]

            clusters.append(tree)
        tm.clusters = clusters
    # print("Warning: maxima iterations reached without convergence.")

    print("=> EM finished")
    topology_list = []
    theta_list = []
    for t in tm.clusters:
        topology_list.append(t.get_topology_array())
        theta_list.append(t.get_theta_array())
    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    # theta_list = np.array(theta_list)

    return loglikelihood, topology_list, theta_list, tm


def main():
    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='EM algorithm for likelihood of a tree GM.')
    parser.add_argument('--sample_filename', type=str, default='data/q_2_5_tm_10node_20sample_4clusters.pkl_samples.txt',
                        help='Specify the name of the sample file (i.e data/example_samples.txt)')
    parser.add_argument('--output_filename', type=str, default='result',
                        help='Specify the name of the output file (i.e data/example_results.txt)')
    parser.add_argument('--num_clusters', type=int, default=4, help='Specify the number of clusters (i.e 3)')
    parser.add_argument('--seed_val', type=int, default=42, help='Specify the seed value for reproducibility (i.e 42)')
    parser.add_argument('--real_values_filename', type=str, default='data/q_2_5_tm_10node_20sample_4clusters.pkl',
                        help='Specify the name of the real values file (i.e data/example_tree_mixture.pkl)')
    # You can add more default parameters if you want.

    print("Hello World!")
    print("This file demonstrates the flow of function templates of question 2.5.")

    print("\n0. Load the parameters from command line.\n")

    args = parser.parse_args()
    print("\tArguments are: ", args)

    simulation = False
    if simulation:
        print("\n1. Make new tree and sample.\n")
        tm_truth = TreeMixture(args.num_clusters, args.num_nodes)
        tm_truth.simulate_pi(seed_val=args.seed_val)
        tm_truth.simulate_trees(seed_val=args.seed_val)
        tm_truth.sample_mixtures(args.num_samples, seed_val=args.seed_val)
    else:
        print("\n1. Load tree from file.\n")
        tm_truth = TreeMixture(0, 0)
        tm_truth.load_mixture(args.real_values_filename)
    samples = tm_truth.samples
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    # print("\n1. Load samples from txt file.\n")
    # samples = np.loadtxt(args.sample_filename, delimiter="\t", dtype=np.int32)
    # num_samples, num_nodes = samples.shape

    print("\n2. Run EM Algorithm.\n")
    loglikelihood, topology_array, theta_array, tm = em_algorithm(args.seed_val, samples, num_clusters=args.num_clusters)

    print("\n3. Save, print and plot the results.\n")
    # save_results(loglikelihood, topology_array, theta_array, args.output_filename)
    for i in range(args.num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    print("\n4. Retrieve real results and compare.\n")
    if args.real_values_filename != "":
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        N = len(samples)
        K = tm_truth.num_clusters
        A = tm_truth.clusters[0].k
        tns = dendropy.TaxonNamespace()
        for k in range(K):
            for j in range(K):
                t_0 = tm.clusters[k]
                t_t = tm_truth.clusters[j]
                t_0.get_tree_newick()
                t_t.get_tree_newick()
                t_0 = dendropy.Tree.get(data=t_0.newick, schema="newick", taxon_namespace=tns)
                t_t = dendropy.Tree.get(data=t_t.newick, schema="newick", taxon_namespace=tns)
                print("\tRF distance result-truth %d - %d:   "%(k,j), dendropy.calculate.treecompare.symmetric_difference(t_0, t_t))


        print("\t4.2. Make the likelihood comparison.\n")
        post = np.ones((N,K)) # posterior p(x^n|T_k, theta_k)
        prior = np.ones(N)
        for n in range(N):
            x = samples[n]
            for k in range(K):
                tree = tm_truth.clusters[k]
                visit_list = [tree.root] # go through whole tree for posterior p(x^n|T_k, theta_k)
                while len(visit_list) != 0:
                    cur_node = visit_list[0]
                    visit_list = visit_list[1:]
                    visit_list = visit_list + cur_node.descendants
                    if cur_node.ancestor: # node with parent
                        # cat[parent][child] = p(child|parent)
                        post[n,k] *= cur_node.cat[x[int(cur_node.ancestor.name)]][x[int(cur_node.name)]] 
                    else: # root node
                        post[n,k] *= cur_node.cat[x[int(cur_node.name)]]
            prior[n] *= np.sum(post[n]*tm_truth.pi) # compute prior p(x^n)
        loglikelihood_truth = np.sum(np.log(prior))
        print("\tLog-Likelihood EM_result vs truth: %d vs %d"%(loglikelihood[-1], loglikelihood_truth))



if __name__ == "__main__":
    main()
