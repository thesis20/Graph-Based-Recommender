import numpy as np
import random
from gensim.models import Word2Vec
from dataloaders import movielens_data as loader


class DeepWalk():

    def __init__(self, window_size, embedding_size, walk_per_vertex,
                 walk_length):
        """
            Parameters:
                window_size (int): size of window when doing SkipGram
                embedding_size (int): dimension to embed in
                walk_per_vertex (int): random walks done for each vertex in
                    the graph.
                walk_length (int): the length of random walks
        """
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_per_vertex = walk_per_vertex
        self.walk_length = walk_length

    def train(self, graph):
        walks = self.do_random_walks(graph)
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(
            walks,
            size=self.embedding_size,
            window=self.window_size,
            min_count=0,  # Ignores words with frequency lower than this
            sg=1,  # Make use of skipgram
            # It is possible to make this multithreaded through the workers
            # parameter
        )
        id2node = dict([(id, node) for id, node in enumerate(graph.nodes())])
        embeddings = np.asarray([model.wv[id2node[i]]
                                 for i in range(len(id2node))])
        return embeddings

    def random_walk(self, node, graph):
        walk = [node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            current_node_neighbors = list(graph.neighbors(current_node))
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


movies, ratings = loader.load_data_ml100k()
graph = loader.generate_bipartite_graph(movies, ratings)

deep_walk = DeepWalk(5, 2, 40, 80)

embeddings = deep_walk.train(graph)
