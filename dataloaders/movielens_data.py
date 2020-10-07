"""Loads data from csv into frames"""

import pandas as pd
import numpy as np
import networkx as nx


def load_data_ml100k():
    """Load the movielens files in and return as pds."""
    movies = pd.read_csv('../data/ml-100k/movies.csv', sep=',')
    ratings = pd.read_csv('../data/ml-100k/ratings.csv', sep=',')

    return movies, ratings


def generate_bipartite_graph(movies_frame, ratings_frame):
    """Convert the movie data into a user-movie biparte graph."""

    full_data = pd.merge(movies_frame, ratings_frame, on='movieId')
    full_data['userId'] = 'u' + full_data['userId'].astype(str)
    full_data['movieId'] = 'm' + full_data['movieId'].astype(str)

    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(full_data.userId, bipartite=0)
    bipartite_graph.add_nodes_from(full_data.movieId, bipartite=1)

    return bipartite_graph


def generate_bipartite_adjencency_matrix(movies_frame, ratings_frame):
    """Convert movie and ratings frame into an adjencency matrix."""

    movie_ids = list(movies_frame.movieId.unique())
    user_ids = list(ratings_frame.userId.unique())

    number_of_movies = len(movie_ids)
    number_of_users = len(user_ids)

    user_movie_adj_matrix = np.zeros((number_of_movies, number_of_users))

    for name, group in ratings_frame.groupby(['userId', 'movieId']):
        user_id, movie_id = name
        user_index = user_ids.index(user_id)
        movie_index = movie_ids.index(movie_id)
        user_movie_adj_matrix[movie_index, user_index] = group[['rating']] \
            .values[0, 0]

    return user_movie_adj_matrix
