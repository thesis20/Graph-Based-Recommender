"""Loads data from csv into frames"""

import pandas as pd
import numpy as np
import networkx as nx


def load_data_ml100k():
    """Load the movielens files in and return as pds."""
    ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t',
                          header=None, index_col=False,
                          names=['userID', 'movieId', 'rating', 'timestamp'])

    return ratings


def read_ml_kfold_splits(folds=5):
    train_test_sets = []
    for i in range(folds):
        train_set = pd.read_csv(
            'data/ml-kfold-splits/' + str(i) + '/train.csv',
            sep=';')
        test_set = pd.read_csv(
            'data/ml-kfold-splits/' + str(i) + '/test.csv',
            sep=';')
        train_test_sets.append((train_set, test_set))

    return train_test_sets


def load_data_yahoo():
    """
    Load the yahoo-movies dataset as dataframe
    """
    ratings = pd.read_csv('../data/yahoo-movies/yahoo-movies.csv', sep=',')
    return ratings


def load_data_comoda():
    """
    Load the CoMoDa dataset as a dataframe
    """
    ratings = pd.read_csv('../data/CoMoDa.csv', sep=';')
    return ratings


def generate_bipartite_graph(ratings_frame, user_column_name,
                             item_column_name):
    """Convert the movie data into a user-movie biparte graph."""

    ratings_frame[user_column_name] = 'u' + \
        ratings_frame[user_column_name].astype(str)
    ratings_frame[item_column_name] = 'i' + \
        ratings_frame[item_column_name].astype(str)

    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(ratings_frame[user_column_name],
                                   bipartite=0)
    bipartite_graph.add_nodes_from(ratings_frame[item_column_name],
                                   bipartite=1)
    edges = list(zip(ratings_frame[user_column_name],
                     ratings_frame[item_column_name]))
    bipartite_graph.add_edges_from(edges)

    return bipartite_graph


def generate_bipartite_adjencency_matrix(movies_frame, ratings_frame):
    """Convert movie and ratings frame into an adjencency matrix."""

    movie_ids = list(movies_frame.itemID.unique())
    user_ids = list(ratings_frame.userID.unique())

    number_of_movies = len(movie_ids)
    number_of_users = len(user_ids)

    user_movie_adj_matrix = np.zeros((number_of_movies, number_of_users))

    for name, group in ratings_frame.groupby(['userID', 'itemID']):
        user_id, movie_id = name
        user_index = user_ids.index(user_id)
        movie_index = movie_ids.index(movie_id)
        user_movie_adj_matrix[movie_index, user_index] = group[['rating']] \
            .values[0, 0]

    return user_movie_adj_matrix
