from Kruskal_v1 import Graph
from Tree import TreeMixture, Tree, Node

import numpy as np
import sys

epsilon = sys.float_info.epsilon


def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100):
    print("Running EM algorithm...")

    # Set threshold for convergence
    THRES = 1e-4
    
    # Set rounds for sieving
    num_sieving = 10

    # Get the dimension of the data
    num_samples = np.size(samples, 0)
    num_nodes = np.size(samples, 1)
    
    # Sieving
    np.random.seed(seed_val)
    seeds = np.random.randint(0, 100000000, num_sieving)
    last_loglikelihoods = []
    tms = []
    for seed in seeds:
        np.random.seed(seed)
        tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
        tm.simulate_pi(seed_val=seed)
        tm.simulate_trees(seed_val=seed)
        tm_loglikelihood, tm = em_helper(tm, samples, num_clusters, max_num_iter=10)
        last_loglikelihoods.append(tm_loglikelihood[-1])
        tms.append(tm)

    # Main procedure for EM algorithm
    print("=> Sieving finished")
    seed = seeds[last_loglikelihoods.index(max(last_loglikelihoods))]
    tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
    tm.simulate_pi(seed_val=seed)
    tm.simulate_trees(seed_val=seed)
    loglikelihood, tm = em_helper(tm, samples, num_clusters, max_num_iter=max_num_iter)

    print("=> EM finished")
    topology_list = []
    theta_list = []
    for t in tm.clusters:
        topology_list.append(t.get_topology_array())
        theta_list.append(t.get_theta_array())
    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)

    return loglikelihood, topology_list, theta_list, tm


def em_helper(tm, samples, num_clusters, max_num_iter=10):
    num_samples = np.size(samples, 0)
    num_nodes = np.size(samples, 1)

    loglikelihood = []
    for iter in range(max_num_iter):
        print("==================== "+str(iter)+"-th iteration ====================")
        # Step 1: Compute the responsibilities
        r = np.ones((num_samples, num_clusters))

        for n, x in enumerate(samples):
            for k, t in enumerate(tm.clusters):
                r[n,k] *= tm.pi[k]
                visit_list = [t.root]
                while len(visit_list) is not 0:
                    cur_node = visit_list[0]
                    visit_list = visit_list[1:]
                    visit_list = visit_list + cur_node.descendants
                    if cur_node.ancestor is None:
                        r[n,k] *= cur_node.cat[x[int(cur_node.name)]]
                    else:
                        r[n,k] *= cur_node.cat[x[int(cur_node.ancestor.name)]][x[int(cur_node.name)]]

        r += epsilon
        marginal = np.reshape(np.sum(r, axis=1), (num_samples, 1))
        loglikelihood.append(np.sum(np.log(marginal)))
        marginal_expand = np.repeat(marginal, num_clusters, axis=1)
        r /= marginal_expand

        # Step 2: Update categorical distribution
        tm.pi = np.mean(r, axis=0)

        # Step 3: Construct directed graphs
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
            edges = np.array(g.maximum_spanning_tree())[:,0:2]
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
        tm.clusters =clusters

    return loglikelihood, tm