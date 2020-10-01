import pandas as pd
import numpy as np
from dataloaders import movielens_data as loader


def get_data():
    movies, ratings = loader.load_data()
    movie_data = pd.merge(ratings, movies, on='movieId')
    user_movie_rating = movie_data.pivot_table(index='userId', columns='title',
                                               values='rating')
    return user_movie_rating


def matrix_factorization(pd_matrix):
    number_of_users = len(pd_matrix)
    number_of_movies = len(pd_matrix.columns)
    features = 20
    steps = 1000
    alpha = 0.0001
    beta = 0.02

    user_features = np.random.rand(number_of_users, features)
    movie_features = np.random.rand(features, number_of_movies)
    numpy_matrix = pd_matrix.to_numpy()

    for step in range(steps):
        for i in range(number_of_users):
            for j in range(number_of_movies):
                if (numpy_matrix[i][j]) > 0:
                    error = numpy_matrix[i][j] - np.dot(user_features[i, :],
                                                        movie_features[:, j])

                    for k in range(features):
                        user_features[i][k] = user_features[i][k] + alpha * \
                                              (2 * error * movie_features[k][j]
                                               - beta * user_features[i][k])
                        movie_features[k][j] = movie_features[k][j] + alpha * (
                                2 * error * user_features[i][k]
                                - beta * movie_features[k][j])

        prediction = np.dot(user_features, movie_features)

        mse = 0
        for i in range(number_of_users):
            for j in range(number_of_movies):
                if (numpy_matrix[i][j]) > 0:
                    mse = mse + pow(
                        numpy_matrix[i][j] - np.dot(user_features[i, :],
                                                    movie_features[:, j]), 2)

                    for k in range(features):
                        mse = (mse + (beta / 2) * (
                                pow(user_features[i][k], 2) +
                                pow(movie_features[k][j], 2)))

        if mse < 0.001:
            break

    return prediction


def main():
    movie_data = get_data()
    recom = matrix_factorization(movie_data)
    return recom


main()
