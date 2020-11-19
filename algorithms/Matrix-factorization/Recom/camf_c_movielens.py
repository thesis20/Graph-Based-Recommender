import pandas as pd
import numpy as np
from math import ceil, sqrt
from sklearn.model_selection import train_test_split


class ContextAwareMatrixFactorization:
    def __init__(self, data, factors):
        self.data = data
        self.factors = factors
        self.unique_users = data['userId'].unique()
        self.unique_items = data['movieId'].unique()
        self.unique_time_of_day = data['timeofday'].unique()
        self.unique_day_of_week = data['dayofweek'].unique()
        self.users_dict = {k: v for v, k in enumerate(self.unique_users)}
        self.items_dict = {k: v for v, k in enumerate(self.unique_items)}
        self.time_of_day_dict = {k: v for v,
                                 k in enumerate(self.unique_time_of_day)}
        self.day_of_week_dict = {k: v for v,
                                 k in enumerate(self.unique_day_of_week)}
        self.number_of_users = len(self.unique_users)
        self.number_of_items = len(self.unique_items)
        self.number_of_time_of_day = len(self.unique_time_of_day)
        self.number_of_day_of_week = len(self.unique_day_of_week)

        self.user_factor_matrix = np.random.uniform(
            0, 1, (self.number_of_users, self.factors))
        self.factor_item_matrix = np.random.uniform(
            0, 1, (self.factors, self.number_of_items))
        self.user_biases = np.random.uniform(size=self.number_of_users)
        self.time_of_day_biases = np.random.uniform(
            size=self.number_of_time_of_day)
        self.day_of_week_biases = np.random.uniform(
            size=self.number_of_day_of_week)
        self.get_average_ratings()

    def get_average_ratings(self):
        """
        Gets the average rating for each item and stores in a dictionary.
        """
        self.average_item_ratings = {}
        for _, row in data.iterrows():
            self.average_item_ratings[row['movieId']] = data.loc[
                self.data['movieId'] == row['movieId']]['rating'].mean()

    def predict_ratings_for_entries(self, df):
        """
        Params:
            df (DataFrame): the dataframe containing rows of actual ratings
                for a specific user that we want predict ratings for
        Returns a dict of
            key: movieId and value: (actual_rating, predicted_rating)
        """
        predictions = {}
        for _, row in df.iterrows():
            user_index = self.users_dict[row['userId']]
            item_index = self.items_dict[row['movieId']]
            time_of_day_index = self.time_of_day_dict[row['timeofday']]
            day_of_week_index = self.day_of_week_dict[row['dayofweek']]
            rating = row['rating']
            predicted_rating = (np.dot(
                self.user_factor_matrix[user_index, :],
                self.factor_item_matrix[:, item_index])
                + self.average_item_ratings[row['movieId']]
                + self.user_biases[user_index]
                + self.time_of_day_biases[time_of_day_index]
                + self.day_of_week_biases[day_of_week_index])
            predictions[row['movieId']] = (rating, predicted_rating)
        return predictions

    def train(self, training_data, lrate, rterm, epochs=1000,
              batch_size=0.001):
        """
        Function that trains the model by learning the parameters
        through stochastic gradient descent
        """
        for epoch in range(epochs):
            batch = training_data.sample(
                ceil(len(training_data.index)*batch_size))

            for _, row in batch.iterrows():
                user_index = self.users_dict[row['userId']]
                item_index = self.items_dict[row['movieId']]
                time_of_day_index = self.time_of_day_dict[row['timeofday']]
                day_of_week_index = self.day_of_week_dict[row['dayofweek']]
                rating = row['rating']

                error = rating - (np.dot(
                    self.user_factor_matrix[user_index, :],
                    self.factor_item_matrix[:, item_index])
                    + self.average_item_ratings[row['movieId']]
                    + self.user_biases[user_index]
                    + self.time_of_day_biases[time_of_day_index]
                    + self.day_of_week_biases[day_of_week_index])

                self.user_biases[user_index] = self.user_biases[
                    user_index] - lrate * (2 * error * (-1) +
                                           2 * rterm *
                                           self.user_biases[user_index])
                self.time_of_day_biases[time_of_day_index] = \
                    self.time_of_day_biases[time_of_day_index] - \
                    lrate * (2 * error * (-1) + 2 * rterm *
                             self.time_of_day_biases[time_of_day_index])
                self.day_of_week_biases[day_of_week_index] = \
                    self.day_of_week_biases[day_of_week_index] - \
                    lrate * (2 * error * (-1) + 2 * rterm *
                             self.day_of_week_biases[day_of_week_index])

                for k in range(self.factors):
                    self.user_factor_matrix[user_index][k] = \
                        self.user_factor_matrix[user_index][k] - lrate * \
                        (2 * error * (-1) *
                         self.factor_item_matrix[k][item_index]
                         + 2*rterm * self.user_factor_matrix[user_index][k])
                    self.factor_item_matrix[k][item_index] = \
                        self.factor_item_matrix[k][item_index] - lrate * (
                        2 * error * (-1) *
                        self.user_factor_matrix[user_index][k] +
                        2 * rterm * self.factor_item_matrix[k][item_index])

            if epoch % 10 == 0:
                mse, mae, rmse = self.calculate_metrics(training_data)
                print("MSE:", mse, "-- RMSE:",
                      rmse, "-- MAE:", mae)

    def calculate_metrics(self, data):
        """
        Calculates the metrics MSE, MAE and RMSE
        Parameter:
            data (DataFrame): the data to calculate the error for
        Returns: mse, mae, rmse
        """
        mse = 0
        mae = 0
        for _, row in data.iterrows():
            user_index_mse = self.users_dict[row['userId']]
            item_index_mse = self.items_dict[row['movieId']]
            time_of_day_index_mse = self.time_of_day_dict[row['timeofday']]
            day_of_week_index_mse = self.day_of_week_dict[row['dayofweek']]
            rating = row['rating']

            predicted_rating = (np.dot(
                self.user_factor_matrix[user_index_mse, :],
                self.factor_item_matrix[:, item_index_mse])
                + self.average_item_ratings[row['movieId']]
                + self.user_biases[user_index_mse]
                + self.time_of_day_biases[time_of_day_index_mse]
                + self.day_of_week_biases[day_of_week_index_mse])
            mse = mse + (rating - predicted_rating)**2
            mae = mae + abs(rating - predicted_rating)

        mse = mse / len(data.index)
        mae = mae / len(data.index)
        rmse = sqrt(mse)

        return mse, mae, rmse


data = pd.read_csv('data/context_ratings.csv',
                   sep=',')
train_set, test_set = train_test_split(data, test_size=0.2)

algo = ContextAwareMatrixFactorization(data, 20)
algo.train(train_set, 0.001, 0.001)
mse, mae, rmse = algo.calculate_metrics(test_set)
print("Errors for the test data", "MSE:", mse, "-- RMSE:",
      rmse, "-- MAE:", mae)
