import networkx as nx
import numpy as np
import numpy.random as nprnd
import random
import math
import pickle
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from util import *

VAL_RATIO, TEST_RATIO = 0.2, 1
GRAPH_MIN_NUM = 10000


""" additive label noise """
def additive_label_noise(args, num_graphs, min_n, max_n, num_colors, graph_type, corr, noise):
    graphs = make_graph(args, num_graphs, min_n, max_n, graph_type)
    new_graphs = []
    for graph in graphs:
        ans, idx = max_node_degree(args, graph)
        graph_nodes = len(graph.nodes)

        colors_ind = nprnd.randint(1, num_colors+1, (graph_nodes, args.node_dim))
        assign_node_features(graph, colors_ind)
        if random.random() < corr:
            if args.sampling == 'gaussian':
                added_noise = nprnd.normal(args.mean, noise)
            elif args.sampling == 'gamma': # noise from zero-mean gamma distribution
                mean = args.mean - 2*noise
                added_noise = nprnd.gamma(2, noise) + mean
            elif args.sampling == 't_dist':
                added_noise = nprnd.standard_t(1) + args.mean
            ans += added_noise

        new_graphs.append((graph, ans))
    return new_graphs

""" instance dependent label noise """
def dependent_label_noise(args, num_graphs, min_n, max_n, num_colors, graph_type, corr, noise):
    graphs = make_graph(args, num_graphs, min_n, max_n, graph_type)
    new_graphs = []
    for graph in graphs:
        ans, idx = max_node_degree(args, graph)
        graph_nodes = len(graph.nodes)

        max_color = nprnd.randint(20, num_colors+1)
        colors_ind = nprnd.randint(1, max_color, (graph_nodes, args.node_dim))

        assign_node_features(graph, colors_ind)
        if random.random() > corr:
            ans = np.max(colors_ind)
            
        new_graphs.append((graph, ans))
    return new_graphs

def assign_node_features(graph, colors_ind):
    node_dict = {}
    ind = 0
    for node in graph.nodes:
        node_dict[node] = colors_ind[ind].tolist() 
        ind += 1
    nx.set_node_attributes(graph, node_dict, 'node_features')

def add_self_loops(graph):
    for n in graph.nodes:
        graph.add_edge(n, n, weight=0)

def max_node_degree(args, graph):
    degree = dict(graph.degree)
    ans = max(degree.values())
    idx = [k for k in degree if degree[k] == ans]
    return ans, idx
    
def generate_random_trees(n):
    return nx.random_tree(n)
    
def generate_random_graphs(n, p):    
    while True:  
        graph = nx.random_graphs.erdos_renyi_graph(n, p)
        if nx.is_connected(graph):
            break 
    return graph

def generate_complete_graphs(n):
    return nx.complete_graph(n)

def sparse_connected_graph(args, n, p, num_components):
    graphs = []
    for i in range(num_components):
        graphs.append(generate_random_graphs(n, p))

    current_graph = graphs[0]
    for i in range(1, num_components):
        current_graph = nx.disjoint_union(current_graph, graphs[i])
        node1 = nprnd.randint((i-1)*n, i*n)
        node2 = nprnd.randint(i*n, (i+1)*n)
        current_graph.add_edge(node1, node2)   

    return current_graph

def load_data(data):
    s2vs = []
    for g, ans in data:
        neighbors = []
        node_features = []
        for i in sorted(list(g.nodes())):
            neighbors.append(list(g.neighbors(i)))
            node_features.append(g.nodes[i]['node_features'])
        node_features = np.array(node_features)
        node_features = torch.from_numpy(node_features).float()
        s2vg = S2VGraph(ans, node_features, neighbors, g)
        s2vs.append((s2vg, ans))
    return s2vs

def generate_graphs_various_nodes(args):    
    if args.min_n == args.max_n:
        n = args.min_n
    else:
        n = nprnd.randint(args.min_n, args.max_n)
    if args.graph_type == 'random_graph':
        graph = generate_random_graphs(n, args.p)
    elif args.graph_type == 'tree':
        graph = generate_random_trees(n)
    elif args.graph_type == 'complete':
        graph = generate_complete_graphs(n)
    elif args.graph_type == 'path':
        graph = nx.path_graph(n)
    elif args.graph_type == 'ladder':
        graph = nx.ladder_graph(n)
    elif args.graph_type == 'tree':
        graph = nx.random_tree(n)
    elif args.graph_type == 'cycle':
        graph = nx.cycle_graph(n)
    elif args.graph_type == 'star':
        graph = nx.star_graph(n)
    elif args.graph_type == '4regular':
        graph = nx.random_regular_graph(4, n)
    else:
        print("Invalid graph type.")

    return graph    

'''
Generate graphs based on parameters.
'''
def make_graph(args, num_graphs, min_n, max_n, graph_type):
    graphs = []
    if graph_type == 'general':
        num_each = int(num_graphs/9)
        args.min_n, args.max_n, args.graph_type = min_n, max_n, 'random_graph'
        for p in np.linspace(0.1, 0.9, 9):
            args.p = p
            for i in range(num_each):
                graph = generate_graphs_various_nodes(args)
                graphs.append(graph)
    elif graph_type == 'expander':
        args.min_n, args.max_n, args.graph_type = min_n, max_n, 'random_graph'
        args.p = 0.8
        for i in range(num_graphs):
            graph = generate_graphs_various_nodes(args)
            graphs.append(graph)
    elif graph_type == 'complete' or graph_type == 'path' or graph_type == 'ladder' or graph_type == 'tree':
        args.min_n, args.max_n, args.graph_type = min_n, max_n, graph_type
        for i in range(num_graphs):
            graph = generate_graphs_various_nodes(args)
            graphs.append(graph)
    elif graph_type == 'cycle' or graph_type == 'star' or graph_type == '4regular':
        args.min_n, args.max_n, args.graph_type = min_n, max_n, graph_type
        for i in range(num_graphs):
            graph = generate_graphs_various_nodes(args)
            graphs.append(graph)
    else:
        print("Invalid graph type!")
        exit()

    return graphs

def augment(dataset, no_aug, goal=GRAPH_MIN_NUM):
    if no_aug:
        return dataset
    if len(dataset) >= goal:
        return dataset
    
    ratio = 1.0*goal/len(dataset)
    i, _ = divmod(ratio, 1)
    i = int(i)
    new_dataset = dataset*i
    
    r = goal - len(new_dataset)

    inds = np.random.choice(len(dataset), r, replace=False)
    r_dataset = [dataset[ind] for ind in inds]
    
    new_dataset.extend(r_dataset)
    return new_dataset


def main():
    # parameters for graph_generation
    parser = argparse.ArgumentParser(description='Graph generation')
    parser.add_argument('--graph_type', type=str, default='random_graph', help='select which graph type to generate')
    parser.add_argument('--train_min_n', default=20, type=int, help='min number of nodes in the graph')
    parser.add_argument('--train_max_n', default=40, type=int, help='max number of nodes in the graph')
    parser.add_argument('--test_min_n', default=50, type=int, help='min number of nodes in the graph')
    parser.add_argument('--test_max_n', default=70, type=int, help='min number of nodes in the graph')
    parser.add_argument('--train_color', default=100, type=int, help='number of colors')
    parser.add_argument('--test_color', default=100, type=int, help='number of colors')
    parser.add_argument('--node_dim', default=1, type=int, help='number of node features')
    parser.add_argument('--train_graph', default='general', type=str, help='train graph type')
    parser.add_argument('--test_graph', default='general', type=str, help='test graph type')
    parser.add_argument('--folder', default='data', type=str, help='run file')
    parser.add_argument('--sampling', default='gaussian', type=str, help='distribution for addtive label noise', choices=['gaussian', 'gamma', 't_dist'])
    parser.add_argument('--mean', default=0, type=float, help='mean of the gaussian distribution in additive noise')
    parser.add_argument('--train_corr', default=0.9, type=float, help='noise ratio in training data')
    parser.add_argument('--train_noise', default=0.0, type=float, help='noise scale in training data')
    parser.add_argument('--test_corr', default=0.0, type=float, help='noise ratio in test data')
    parser.add_argument('--test_noise', default=0.0, type=float, help='noise scale in test data')

    parser.add_argument('--no_aug', action='store_true', default=False, help='disables augmenting dataset')
    parser.add_argument('--num_graphs', default=10000, type=int, help='num of graphs we want in the train dataset')
    parser.add_argument('--run', type=str, help='run file')
    parser.add_argument('--prefix', default='additive', choices=['dependent', 'additive'], type=str, help='task prefix')
    args = parser.parse_args()
    Path(args.folder).mkdir(parents=True, exist_ok=True)
    output = './%s/%s_dim%s_sampling:%s_mean%s_graph:%s_N%s_%s_C%s_test_G%s_N%s_%s_C%s_corr_%s_%s_noise_%s_%s_effective_number_%s_noaug%s.pickle' %(args.folder, args.prefix, args.node_dim, args.sampling, args.mean, args.train_graph, args.train_min_n, args.train_max_n, args.train_color, args.test_graph, args.test_min_n, args.test_max_n, args.test_color, args.train_corr, args.test_corr, args.train_noise, args.test_noise, args.num_graphs, args.no_aug)
    
    if not args.run:
        if prefix == 'additive':
            args.run = f"{prefix}_{sampling}_{mean}_{train_noise}_corr{train_corr}"
        else:
            args.run = f"{prefix}_corr{train_corr}"
    
    gen_funcs = {'dependent': dependent_label_noise, 'additive': additive_label_noise}
    
    data_gen = gen_funcs[args.prefix]
    
    train = data_gen(args, args.num_graphs, args.train_min_n, args.train_max_n, args.train_color, args.train_graph, args.train_corr, args.train_noise)
    train = augment(train, args.no_aug)
    train = load_data(train)
    val = data_gen(args, max(int(args.num_graphs*VAL_RATIO),1), args.train_min_n, args.train_max_n, args.train_color, args.train_graph, args.train_corr, args.train_noise)
    val = augment(val, args.no_aug, int(GRAPH_MIN_NUM*VAL_RATIO))
    val = load_data(val)
    test = data_gen(args, max(int(args.num_graphs*TEST_RATIO),1),  args.test_min_n, args.test_max_n, args.test_color, args.test_graph, args.test_corr, args.test_noise)
    test = augment(test, args.no_aug, goal=int(GRAPH_MIN_NUM*TEST_RATIO))
    test = load_data(test)
    
    filename = "./run/%s.txt" %(args.run)
    file = open(filename, "w+") 
    file.write("%s\n" %(output))
    file.close()
    
    with open(output, 'wb') as f:
        pickle.dump((train, val, test), f)

    print("data saved to %s" % output)
    
if __name__ == '__main__':
    main()