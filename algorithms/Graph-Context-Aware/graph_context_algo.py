import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from operator import itemgetter


def load_data():
    """Load the movielens files in and return as pds."""
    movies = pd.read_csv('../../data/ml-100k/movies.csv', sep=',')
    ratings = pd.read_csv('../../data/ml-100k/ratings.csv', sep=',')

    return movies, ratings


def split_items():
    movies, ratings = load_data()

    users_distinct = ratings.userId.unique()
    items_distinct = ratings.movieId.unique()

    # hours = range(24)
    month = range(1, 13)
    weekday = range(7)

    context_product = list(product(items_distinct, weekday, month))  # Nc
    context_product_dict = {k: v for v, k in enumerate(context_product)}

    item_split_matrix = np.zeros((len(users_distinct),
                                  len(list(context_product))),
                                 dtype=np.dtype('float32'))  # T

    for item in ratings.itertuples():
        loaded_date = datetime.utcfromtimestamp(item[4])
        weekday = loaded_date.weekday()
        # hour = loaded_date.hour
        month = loaded_date.month
        # timestamp = (weekday, month)
        userId = item[1]
        movieId = item[2]
        rating = item[3]

        idx = context_product_dict[(movieId, weekday, month)]
        item_split_matrix[userId-1][idx] = rating/5

    return context_product, item_split_matrix


def generate_graph(items, context_product):
    # Similarity calculation

    items_t = np.transpose(items)

    user_similarity_len_2 = np.dot(items, items_t)
    # item_similarity = np.dot(matrix_2, matrix_1)

    user_similarity_len_4 = np.dot(user_similarity_len_2,
                                   user_similarity_len_2)

    return user_similarity_len_4


def knn_rec(user_id, k, user_similarity, item_split_matrix, context_product, context):
    user_row = user_similarity[user_id-1]
    most_similar_users = (user_row.argsort()[-k-1::][::-1])[1:]

    recs = {}

    item_split_row = item_split_matrix[user_id-1]
    not_rated_indices = np.argwhere(item_split_row == 0.0)

    for item in not_rated_indices:
        summation = 0
        counter = 0
        for sim_user in most_similar_users:
            if item_split_matrix[sim_user][item[0]] > 0:
                summation += item_split_matrix[sim_user][item[0]]
                counter += 1

        if counter == 0:
            continue
        else:
            recs[item[0]] = summation/counter

    res = dict(sorted(recs.items(), key=itemgetter(1), reverse=True)[:k])

    movie_info, _ = load_data()

    movie_titles = []

    for key, value in res.items():
        movie_titles.append(get_movie_name(context_product[key][0], movie_info))

    return movie_titles


def get_movie_name(movieId, movies):
    """Find the movie name based on id."""
    return movies.loc[movies['movieId'] == int(movieId)]['title'].iloc[0]


if __name__ == '__main__':
    context_product, items = split_items()
    user_similarity = generate_graph(items, context_product)
    res = knn_rec(1, 10, user_similarity, items, context_product, (5, 12))

    for result in res:
        print(result)
