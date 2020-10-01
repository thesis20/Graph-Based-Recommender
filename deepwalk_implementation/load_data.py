import pandas as pd
import numpy as np


def read_data():
    movie_data = pd.read_csv('../dataset/ml-latest-small/movies.csv')

    rating_data = pd.read_csv('../dataset/ml-latest-small/ratings.csv')

    merged_table = pd.merge(movie_data, rating_data, on='movieId')

    edge_list = []
    for index, row in merged_table.iterrows():
        edge_list.append((row['movieId'], row['userId']))
        

    return edge_list




