import numpy as np
import random

# import networkx as nx
# from gensim.models import Word2Vec, KeyedVectors
from dataloaders import movielens_data as loader


class DeepWalk():

    def __init__(self, window_size, embedding_size, walk_per_vertex,
                 walk_length):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_per_vertex = walk_per_vertex
        self.walk_length = walk_length

    def train(self, graph):
        walks = self.do_random_walks(graph)
        walks = [[str(node) for node in walk] for walk in walks]

    def random_walk(self, node, graph):
        walk = [node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            current_node_neighbors = graph.neighbors(current_node)
            if len(current_node_neighbors) == 0:
                break
            else:
                index = int(np.floor(np.random.rand()
                                     * len(current_node_neighbors)))
                walk.append(current_node_neighbors[index])
        return walk

    def do_random_walks(self, graph):
        walks = []
        all_nodes = list(graph.nodes())

        for walk_count in range(self.walk_per_vertex):
            random.shuffle(all_nodes)
            for node in all_nodes:
                walks.append(self.random_walk(node, graph))
        return walks


""" 
    def build_binary_tree(self, vertices):
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

        self.calculate_tree_depth(nodes[0])
        return nodes.pop(), leaves


    def calculate_tree_depth(self, node, depth=1):
        node.depth = depth
        if node.left is not None:
            calculate_tree_depth(node.left, depth + 1)
        if node.right is not None:
            calculate_tree_depth(node.right, depth + 1)
 """

graph = loader.generate_bipartite_graph(loader.load_data())

deep_walk = DeepWalk()

deep_walk.train(graph)
