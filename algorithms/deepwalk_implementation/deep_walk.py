import numpy as np
import random
import pandas as pd
from gensim.models import Word2Vec
from dataloaders import movielens_data as loader
# from dataloaders.read_comoda_kfold_splits import read_comoda_kfold_splits
from algorithms.evaluation.precision_recall import precision_recall_at_k
from algorithms.evaluation.mapk import calculate_map
from algorithms.evaluation.ndcg import calculate_ndcg
from scipy import spatial
from math import sqrt
from datetime import datetime
from dataloaders.read_yahoo_kfold_splits import read_yahoo_kfold_splits


class DeepWalk():

    def __init__(self, window_size, embedding_size, walk_per_vertex,
                 walk_length, seed=42):
        """
            Parameters:
                window_size (int): size of window when doing SkipGram
                embedding_size (int): dimension to embed in
                walk_per_vertex (int): random walks done for each vertex in
                    the graph.
                walk_length (int): the length of random walks
                seed (int): seed for the random generator
        """
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_per_vertex = walk_per_vertex
        self.walk_length = walk_length
        random.seed(seed)

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
        self.id2node = dict([(id, node) for id, node in enumerate(
            graph.nodes())])
        self.node2id = dict([(node, id) for id, node in enumerate(
            graph.nodes())])
        self.embeddings = np.asarray([model.wv[self.id2node[i]]
                                      for i in range(len(self.id2node))])
        self.embedding_tree = spatial.KDTree(self.embeddings)
        self.user_embedding_ids = [key for key, value
                                   in self.id2node.items()
                                   if str.startswith(value, 'u')]
        self.user_embedding_tree = spatial.KDTree(
            self.embeddings[self.user_embedding_ids])
        self.amount_of_users = len(
            [x for x in graph.nodes() if str.startswith(x, 'u')])
        return self.embeddings

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

    def find_nearest_neighbors(self, embedding_id, k=1):
        # Recom tuple is (distances, embedding_indexes)
        recom_tuple = self.embedding_tree.query(
            self.embeddings[embedding_id], k + 1)
        # Remove the first similar user as that is the user itself
        # We only look at the indices (second part of the tuple)
        indices = np.delete(recom_tuple[1], 0)
        # Look up nodes
        recom_movies = list([int(deep_walk.id2node[x][1:]) for x in indices])
        return recom_movies

    def find_nearest_users(self, embedding_id, k=1):
        # Recom tuple is (distances, embedding_indexes)
        recom_tuple = self.user_embedding_tree.query(
            self.embeddings[embedding_id], k + 1)
        # Remove the first similar user as that is the user itself
        # We only look at the indices (second part of the tuple)
        indices = np.delete(recom_tuple[1], 0)
        # Find the user nodes
        recom_users = list([int(deep_walk.id2node[x][1:]) for x in indices])
        return recom_users

    def get_movie_titles(self, embedding_indices):
        movies_df = pd.read_csv('../data/ml-100k/movies.csv', sep=',')
        recommendations = []
        for index in embedding_indices:
            if str.startswith(self.id2node[index], 'i'):
                itemID = str.replace(self.id2node[index], 'i', '')
                recommendations.append(movies_df.loc[
                    movies_df['movieId'] == int(itemID), 'title'].values[0])
            else:
                recommendations.append(self.id2node[index])

    def get_recommendation_for_user(self, userID):
        embedding_id = self.node2id.get('u' + str(userID))

        if embedding_id is not None:
            return self.find_nearest_users(embedding_id, 10)
        return []

    def calculate_metrics(self, test_data, ratings_dict):
        """
        Parameters:
        ----------------
        test_data (dataframe): the test data
        ratings_dict (dict): key is (userId, itemId) pair,
            value is the rating
        ----------------
        Returns: mae, rmse, precision, recall, mapk, ndcg
        """
        mae = 0
        mse = 0
        hits = 0
        predictions = []
        relevant_items = {}
        users_top_k_dict = {}
        for index, row in test_data.iterrows():
            userId = row['userId']
            itemId = row['movieId']
            real_rating = row['rating']
            recom_users = self.get_recommendation_for_user(
                int(userId))
            if users_top_k_dict.get(userId) is None:
                users_top_k_dict[userId] = self.get_top_k_items(
                    recom_users, test_data)
            similar_users_ratings = []
            for user in recom_users:
                rating = ratings_dict.get(('u' + str(user),
                                           'i' + str(itemId)))
                if rating is not None:
                    similar_users_ratings.append(rating)

            if len(similar_users_ratings) != 0:
                hits += 1
                prediction = np.mean(similar_users_ratings)
                mae = mae + abs(prediction - real_rating)
                mse = mse + (prediction - real_rating)**2
                predictions.append((userId, itemId,
                                    real_rating, prediction, None))
            similar_users_ratings = []
            if row['rating'] >= 3:
                if relevant_items.get(userId) is None:
                    relevant_items[userId] = [itemId]
                else:
                    relevant_items[userId].append(itemId)
        mae = mae / hits
        mse = mse / hits
        rmse = sqrt(mse)
        precision, recall = precision_recall_at_k(predictions)
        mapk = calculate_map(users_top_k_dict, relevant_items, 10)
        ndcg = calculate_ndcg(users_top_k_dict, relevant_items)

        return mae, rmse, precision, recall, mapk, ndcg

    def get_top_k_items(self, similar_users, test_data, k=10):
        item_ratings = {}

        for user in similar_users:
            user_data = test_data.loc[test_data['userId'] == user]
            for _, row in user_data.iterrows():
                if item_ratings.get(row['movieId']) is None:
                    item_ratings[row['movieId']] = [row['rating']]
                else:
                    item_ratings[row['movieId']].append(row['rating'])
        for key, value in item_ratings.items():
            item_ratings[key] = np.mean(value)

        top_items = sorted(item_ratings,
                           key=item_ratings.get, reverse=True)[:k]
        return top_items


# Comoda
# train_test_splits = read_comoda_kfold_splits()

# yahoo
train_test_splits = read_yahoo_kfold_splits()

maes = []
rmses = []
precisions = []
recalls = []
mapks = []
ndcgs = []
for train, test in train_test_splits:
    graph = loader.generate_bipartite_graph(train, 'userId', 'movieId')

    ratings_dict = dict([((row['userId'], row['movieId']),
                          row['rating']) for _, row in train.iterrows()])

    start = datetime.now()
    print('Doing the DeepWalk...')
    print(start)
    deep_walk = DeepWalk(window_size=5, embedding_size=20, walk_per_vertex=40,
                         walk_length=80)
    deep_walk.train(graph)
    print('Done with the DeepWalk...')
    print(datetime.now() - start)

    mae, rmse, precision, recall, mapk, ndcg = deep_walk.calculate_metrics(
        test, ratings_dict)
    print('mae', mae, 'rmse', rmse)
    print('precision', precision, 'recall', recall, 'mapk',
          mapk, 'ndcg', ndcg)
    maes.append(mae)
    rmses.append(rmse)
    precisions.append(precision)
    recalls.append(recall)
    mapks.append(mapk)
    ndcgs.append(ndcg)

print('MEANS', 'mae', np.mean(maes),
      'rmse', np.mean(rmses))
print('MEANS', 'precision', np.mean(precisions),
      'recall', np.mean(recalls), 'mapk', np.mean(mapks),
      'ndcg', np.mean(ndcgs))
