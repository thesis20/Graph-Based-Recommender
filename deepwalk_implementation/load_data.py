import pandas as pd


def read_data():
    # ml-latest-small must be in the same folder as the program
    movie_data = pd.read_csv('ml-latest-small/movies.csv')

    rating_data = pd.read_csv('ml-latest-small/ratings.csv')

    merged_table = pd.merge(movie_data, rating_data, on='movieId')

    edge_list = []
    for index, row in merged_table.iterrows():
        edge_list.append((row['movieId'], row['userId']))

    return edge_list
