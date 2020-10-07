import pandas as pd
import numpy as np
from itertools import product
from operator import itemgetter


def load_data():
    """
    Load the movielens data and transform the timestamp into month and weekday
    for context dimensions.

    Returns
    -------
    movies : Pandas dataframe
        Contains information about movies.
    ratings : Pandas dataframe
        Contains information about which users rated which movies.

    """
    movies = pd.read_csv('../../data/ml-100k/movies.csv', sep=',')
    ratings = pd.read_csv('../../data/ml-100k/ratings.csv', sep=',')

    ratings['timestamp'] = (pd.to_datetime(ratings['timestamp'], unit='s'))
    ratings['month'] = pd.DatetimeIndex(ratings['timestamp']).month
    ratings['weekday'] = pd.DatetimeIndex(ratings['timestamp']).weekday

    return movies, ratings


def split_items():
    """
    Splits the data and creates fictive items for each context dimension of
    size (context1 X .. contextN X items)


    Returns
    -------
    context_product : List of tuples (movie_id, weekday, month)
        Movie_id from dataset, weekday from 0-6, month from 1-12.
    item_split_matrix : ndarray
        Normalized ratings for fictive items based on split on context
        dimensions.

    """
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
        weekday = item[6]
        month = item[5]
        userId = item[1]
        movieId = item[2]
        rating = item[3]

        idx = context_product_dict[(movieId, weekday, month)]
        item_split_matrix[userId-1][idx] = rating/5

    return context_product, item_split_matrix


def generate_user_similarity(item_split_matrix, context_product, path_length):
    """
    Compute user similarity with a given path length, path length must be an
    even number of 2 or higher.

    Parameters
    ----------
    item_split_matrix : ndarray
        Normalized ratings for fictive items based on split on context
        dimensions.
    context_product : List of tuples (movie_id, weekday, month)
        Movie_id from dataset, weekday from 0-6, month from 1-12.
        Contains all possible combinations of movies and contexts.
    path_length : integer
        The amount of steps to take while searching for similiar users.

    Returns
    -------
    res : Array of float32
        A symmetrical matrix of users and their similarity.

    """

    # Similarity calculation
    items_t = np.transpose(item_split_matrix)

    user_similarity_len_2 = np.dot(item_split_matrix, items_t)
    # item_similarity = np.dot(items_t, items)

    res = user_similarity_len_2

    steps_taken = 0
    while steps_taken < path_length - 2:
        res = np.dot(user_similarity_len_2, res)
        steps_taken += 2

    return res


def knn_rec(user_id, k, user_similarity, item_split_matrix, context_product,
            context):
    """
    Generate top k recommendations based on kNN algorithm.

    Parameters
    ----------
    user_id : int
        Id of the user.
    k : int
        Amount of recommendations to be generated.
    user_similarity : ndarray
        Symmetrical matrix that defines the similarity of users.
    item_split_matrix : ndarray
        Normalized ratings for fictive items based on split on context
        dimensions.
    context_product : List of tuples (movie_id, weekday, month)
        Movie_id from dataset, weekday from 0-6, month from 1-12.
        Contains all possible combinations of movies and contexts.
    context : tuple (weekday, month)
        The tuple about the users current context, in this case the weekday
        and month for the recommendation.

    Returns
    -------
    List of str
        Returns a list of top k movie titles recommended for the user.

    """
    user_row = user_similarity[user_id-1]
    most_similar_users = (user_row.argsort()[-k-1::][::-1])[1:]

    recs = {}

    item_split_row = item_split_matrix[user_id-1]
    not_rated_indices = np.argwhere(item_split_row == 0.0)

    for not_rated_item in not_rated_indices:
        summation = 0
        counter = 0
        for sim_user in most_similar_users:
            if item_split_matrix[sim_user][not_rated_item[0]] > 0:
                summation += item_split_matrix[sim_user][not_rated_item[0]]
                counter += 1

        if counter == 0:
            continue
        else:
            recs[not_rated_item[0]] = (context_product[not_rated_item[0]],
                                       summation/counter)

    movie_info, _ = load_data()
    movie_titles = []

    # If the filter did not find k recommendations, we expand the first context
    # dimension to include next entry in the dimension (wednesday -> thursday).
    i = 0
    y = 0
    while len(movie_titles) < k:
        if i % 7 == 6:
            y += 1
            if y > 12:
                break

        context_recs = {k: v for k, v in recs.items() if (v[0][1]+i % 7,
                                                          v[0][2] + y % 12)
                        == context}

        res = dict(sorted(context_recs.items(), key=itemgetter(1),
                          reverse=True))

        for key, value in res.items():
            title = get_movie_name(context_product[key][0], movie_info)
            if title not in movie_titles:
                movie_titles.append(title)
        i += 1

    return movie_titles[:k]


def get_movie_name(movieId, movies):
    """
    Find a movie name based on Id

    Parameters
    ----------
    movieId : integer
        The id of the movie you want the name of.
    movies : Pandas dataframe
        The dataframe containing information about movies.

    Returns
    -------
    string
        The title of the movie.

    """
    return movies.loc[movies['movieId'] == int(movieId)]['title'].iloc[0]


if __name__ == '__main__':
    context_product, item_split_matrix = split_items()

    user_similarity = generate_user_similarity(item_split_matrix,
                                               context_product, 2)

    res = knn_rec(558, 10, user_similarity, item_split_matrix, context_product,
                  (5, 12))

    for item in res:
        print(item)
