import numpy as np
from math import e
from load_data import read_data
import os.path
import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools
import sys
from binary_tree import Node

def deep_walk(edge_list, window_size, embedding_size, wpv, wl, lr=0.025):
    g = nx.Graph()
    g.add_edges_from(edge_list)
    
    adjacency_dict_movie, adjacency_dict_user = make_adjacency_list(edge_list)

    all_vertices = []
    all_vertices = list(zip(adjacency_dict_movie.keys(), ['m'] * len(adjacency_dict_movie)))
    all_vertices += list(zip(adjacency_dict_user.keys(), ['u'] * len(adjacency_dict_user)))

    binary_tree_root, leaves = build_binary_tree(all_vertices)

    phi = np.random.uniform(size=(len(all_vertices), embedding_size))

    if not os.path.exists('output.txt'):
        f = open('output.txt', 'w')
        for _ in range(len(all_vertices)):
            for _ in range(embedding_size):
                f.write(str(random.random()) + ' ')
            f.write('\n')
        f.close()

    
    for _ in range(wpv):
        all_vertices_shuffled = all_vertices.copy()
        random.shuffle(all_vertices_shuffled)
        for v in all_vertices_shuffled:
            rw = random_walk(adjacency_dict_user, adjacency_dict_movie, v, wl)
            prob = skip_gram(rw, window_size, phi, all_vertices, leaves)
            loss = -np.log(prob)

            
                

def skip_gram(random_walk, window_size, phi, all_vertices, leaves):
    for index, v in enumerate(random_walk):
        one_hot = np.zeros(phi.shape[0])
        one_hot[all_vertices.index(v)] = 1
        row_rep = np.dot(one_hot, phi)
        prob = 1.0

        for u in range(max(0,index - window_size), min(index + window_size, len(random_walk))):
            # find leaf for the vertex
            node = next(x for x in leaves if x.data == random_walk[u])

            while node.parent is not None:
                parent = node.parent
                if parent.left is node:
                    prob = prob * sigmoid(parent.probability[0] * row_rep)
                else:
                    prob = prob * sigmoid(parent.probability[1] * row_rep)
                node = parent
    return prob
                
            


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_node_n(root, j):
    li=[root]
    while(root!=1):
        root = root//2
        li.append(root)

    li.reverse()
    
    return li[j]

"""
def get_vector(index):
    file = open("output.txt")
    for i, line in enumerate(file):
        if i == index:
            linesplit = line.split()
            return np.asarray(linesplit, dtype=float)
"""

def random_walk(adjacency_dict_user, adjacency_dict_movie, v, walk_length):
    if v[1] == 'u':
        neighbours = adjacency_dict_user[v[0]]
    else:
        neighbours = adjacency_dict_movie[v[0]]

    if walk_length == 0:
        return (v,)
    else:
        return (v,) + random_walk(adjacency_dict_user, adjacency_dict_movie, random.choice(neighbours), walk_length - 1)


def make_adjacency_list(edge_list):
    adjacency_dict_movie = {}
    adjacency_dict_user = {}

    for e in edge_list:
        if e[0] not in adjacency_dict_movie:
            adjacency_dict_movie[e[0]] = [(e[1], 'u')]
        else:
            adjacency_dict_movie[e[0]].append((e[1], 'u'))
        
        if e[1] not in adjacency_dict_user:
            adjacency_dict_user[e[1]] = [(e[0], 'm')]
        else:
            adjacency_dict_user[e[1]].append((e[0], 'm'))

    return adjacency_dict_movie, adjacency_dict_user


def build_binary_tree(vertices):
    nodes = []
    for v in vertices:
        nodes.append(Node(v))   
    leaves = nodes.copy()

    while len(nodes) > 1:
        left = nodes.pop(0)
        right = nodes.pop(0)
        probability = np.random.uniform()
        node = Node(None, (probability, 1 - probability), left, right)
        left.parent = node
        right.parent = node
        nodes.append(node)

    calculate_tree_depth(nodes[0])    
    return nodes.pop(), leaves
            
def calculate_tree_depth(node, depth = 1):
    node.depth = depth
    if node.left is not None:
        calculate_tree_depth(node.left, depth + 1)
    if node.right is not None:
        calculate_tree_depth(node.right, depth + 1)

edge_list = read_data()

deep_walk(edge_list, 2, 60, 5, 5)