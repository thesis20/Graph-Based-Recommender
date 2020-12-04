import numpy as np
import itertools
from math import ceil, sqrt
from algorithms.evaluation.mapk import calculate_map
from algorithms.evaluation.precision_recall import precision_recall_at_k
from algorithms.evaluation.ndcg import calculate_ndcg
# from dataloaders.movielens_data import read_ml_kfold_splits
# from dataloaders.read_comoda_kfold_splits import read_comoda_kfold_splits
from dataloaders.read_yahoo_kfold_splits import read_yahoo_kfold_splits


class ContextAwareMatrixFactorization:
    def __init__(self, data, user_column_name, item_column_name,
                 rating_column_name, context_column_names, features,
                 average_item_ratings=None):
        """
        Constructor for the class

        Parameters
        ----------
            data (DataFrame): the data for the algorithm to use
            user_column_name (string): Name of the user column
            item_column_name (string): Name of the item column
            rating_column_name (string): Name of the rating column
            context_column_name (list(string)): Names of the context columns.
            features (int): The number of features used in the factorization.
            average_item_ratings (dict(itemId,rating)): average rating of items
        ---------
        """
        self.data = data
        self.user_column_name = user_column_name
        self.item_column_name = item_column_name
        self.rating_column_name = rating_column_name
        self.context_column_names = context_column_names
        self.features = features
        self.unique_users = self.data[self.user_column_name].unique()
        self.unique_items = self.data[self.item_column_name].unique()
        self.unique_contexts = dict()
        for context in self.context_column_names:
            self.unique_contexts[context] = list(
                itertools.product(self.data[context].unique(),
                                  self.unique_items))
        self.users_dict = {k: v for v, k in enumerate(self.unique_users)}
        self.items_dict = {k: v for v, k in enumerate(self.unique_items)}
        self.context_dicts = dict()
        for key, value in self.unique_contexts.items():
            self.context_dicts[key] = {k: v for v, k in enumerate(value)}
        self.number_of_users = len(self.unique_users)
        self.number_of_items = len(self.unique_items)
        self.number_of_contexts = dict()
        for key, value in self.unique_contexts.items():
            self.number_of_contexts[key] = len(value)

        # Set random seed
        np.random.seed(42)
        self.user_feature_matrix = np.random.uniform(
            0, 1, (self.number_of_users, self.features))
        self.feature_item_matrix = np.random.uniform(
            0, 1, (self.features, self.number_of_items))
        self.user_biases = np.random.uniform(size=self.number_of_users)
        self.context_biases = dict()
        for key, value in self.number_of_contexts.items():
            self.context_biases[key] = np.random.uniform(size=value)

        self.average_item_ratings = average_item_ratings
        if self.average_item_ratings is None:
            self.get_average_ratings()

    def get_average_ratings(self):
        """
        Gets the average rating for each item and stores in a dictionary.
        """
        self.average_item_ratings = {}
        for _, row in self.data.iterrows():
            self.average_item_ratings[row[self.item_column_name]] = \
                self.data.loc[self.data[self.item_column_name] == row[
                    self.item_column_name]][self.rating_column_name].mean()
        return self.average_item_ratings

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
            user_index = self.users_dict[row[self.user_column_name]]
            item_index = self.items_dict[row[self.item_column_name]]
            context_indexes = dict()
            for key, value in self.context_dicts.items():
                context_indexes[key] = value[(row[key],
                                              row[self.item_column_name])]
            rating = row[self.rating_column_name]

            predicted_rating = self.predict_rating(user_index, item_index,
                                                   row[self.item_column_name],
                                                   context_indexes)
            predictions[row[self.item_column_name]] = (rating,
                                                       predicted_rating)
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
                user_index = self.users_dict[row[self.user_column_name]]
                item_index = self.items_dict[row[self.item_column_name]]
                context_indexes = dict()
                for key, value in self.context_dicts.items():
                    context_indexes[key] = value[(row[key],
                                                  row[self.item_column_name])]
                rating = row[self.rating_column_name]

                predicted_rating = self.predict_rating(
                    user_index, item_index,
                    row[self.item_column_name],
                    context_indexes)
                error = rating - predicted_rating

                self.user_biases[user_index] = self.user_biases[
                    user_index] - lrate * (2 * error * (-1) +
                                           2 * rterm *
                                           self.user_biases[user_index])
                # Update the context variables
                for key, value in self.context_biases.items():
                    value[context_indexes[key]] = value[context_indexes[key]] \
                        - lrate * (2 * error * (-1) + 2 *
                                   rterm * value[context_indexes[key]])

                for k in range(self.features):
                    self.user_feature_matrix[user_index][k] = \
                        self.user_feature_matrix[user_index][k] - lrate * (
                        2 * error * (-1)
                        * self.feature_item_matrix[k][item_index] + 2
                        * rterm * self.user_feature_matrix[user_index][k])
                    self.feature_item_matrix[k][item_index] = \
                        self.feature_item_matrix[k][item_index] - lrate * (
                        2 * error * (-1)
                        * self.user_feature_matrix[user_index][k] + 2
                        * rterm * self.feature_item_matrix[k][item_index])

            if epoch % 100 == 0:
                mse, mae, rmse = self.calculate_metrics(training_data)
                print('Epoch', epoch, "-- MSE:", mse,
                      "-- RMSE:", rmse, "-- MAE:", mae)

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
            user_index = self.users_dict[row[self.user_column_name]]
            item_index = self.items_dict[row[self.item_column_name]]
            context_indexes = dict()
            for key, value in self.context_dicts.items():
                context_indexes[key] = value[(row[key],
                                              row[self.item_column_name])]
            rating = row[self.rating_column_name]

            predicted_rating = self.predict_rating(user_index, item_index,
                                                   row[self.item_column_name],
                                                   context_indexes)
            mse = mse + (rating - predicted_rating)**2
            mae = mae + abs(rating - predicted_rating)

        mse = mse / len(data.index)
        mae = mae / len(data.index)
        rmse = sqrt(mse)

        return mse, mae, rmse

    def calculate_precision_recall_mapk(self, test_data, k_val=10):
        """
        Parameters:
        ---------------------
        test_data (dataFrame): the test data
        k_val (int): the value of k used for the map at k calculation
        ---------------------
        Returns:
        precision, recall, mapk, ndcg
        """
        # Relevant items and user_top_k_items are used
        # for mapk calculation
        relevant_items = {}
        users_top_k_items = {}
        # predictions used for precision and recall
        predictions = []

        for _, row in test_data.iterrows():
            userId = row[self.user_column_name]
            itemId = row[self.item_column_name]
            unique_test_items = test_data[self.item_column_name].unique()

            # Get top k items
            if users_top_k_items.get(userId) is None:
                users_top_k_items[userId] = self.get_top_k_pred(
                    userId, unique_test_items, k_val)

            context_indexes = {}
            for key, value in self.context_dicts.items():
                context_indexes[key] = value[(row[key], itemId)]
            prediction = self.predict_rating(self.users_dict[userId],
                                             self.items_dict[itemId], itemId,
                                             context_indexes)
            predictions.append((userId, itemId, row[self.rating_column_name],
                                prediction, None))

            if row[self.rating_column_name] >= 3:
                if relevant_items.get(userId) is None:
                    relevant_items[userId] = [itemId]
                else:
                    relevant_items[userId].append(itemId)

        mapk = calculate_map(users_top_k_items, relevant_items, k_val)
        precision, recall = precision_recall_at_k(predictions)
        ndcg = calculate_ndcg(users_top_k_items, relevant_items)

        return precision, recall, mapk, ndcg

    def get_top_k_pred(self, userId, unique_test_items, k_val):
        predictions = {}
        lengths = list([(column_name, len(self.data[column_name].unique()))
                        for column_name in self.context_column_names])
        context_values_dict = {}
        for name in self.context_column_names:
            context_values_dict[name] = self.data[name].unique()
        counters = [0] * len(self.context_column_names)

        def nested_for_context(item_id, counters, lengths, level=0):
            if level == len(counters):
                user_index = self.users_dict[userId]
                item_index = self.items_dict[item_id]
                context_values = dict()
                for index, value in enumerate(lengths):
                    context_values[value[0]] = context_values_dict[
                        value[0]][counters[index]]
                context_indexes = dict([(key,
                                         self.context_dicts[key][(value,
                                                                  item_id)])
                                        for key, value
                                        in context_values.items()])
                predicted_rating = self.predict_rating(user_index, item_index,
                                                       item_id,
                                                       context_indexes)

                predictions[(item_id,) +
                            tuple(context_values.items())] = predicted_rating
            else:
                for _ in range(lengths[level][1]):
                    nested_for_context(item_id, counters, lengths, level + 1)
                    counters[level] += 1
                counters[level] = 0

        for item in unique_test_items:
            nested_for_context(item, counters, lengths, 0)

        sorted_predictions = sorted(predictions,
                                    key=predictions.get, reverse=True)
        top_items = [x[0] for x in sorted_predictions]
        top_items = list(dict.fromkeys(top_items))
        return top_items[:k_val]

    def predict_rating(self, user_index, item_index, item_id, context_indexes):
        predicted_rating = (np.dot(
            self.user_feature_matrix[user_index, :],
            self.feature_item_matrix[:, item_index])
            + self.average_item_ratings[item_id]
            + self.user_biases[user_index]
            + sum([value[context_indexes[key]]
                   for key, value in self.context_biases.items()]))
        return predicted_rating


# movielens
# train_test_sets = read_ml_kfold_splits()

# yahoo
train_test_sets = read_yahoo_kfold_splits()

# CoMoDa
# train_test_sets = read_comoda_kfold_splits()

maes = []
mses = []
rmses = []
precisions = []
recalls = []
mapks = []
ndcgs = []
for train, test in train_test_sets:
    full_data = train.append(test)
    algo = ContextAwareMatrixFactorization(full_data, 'userId', 'movieId',
                                           'rating',
                                           ['gender', 'age_group'], 10)
    algo.train(training_data=train, lrate=0.01, rterm=0.2,
               batch_size=0.01)
    mse, mae, rmse = algo.calculate_metrics(test)
    precision, recall, mapk, ndcg = algo.calculate_precision_recall_mapk_ndcg(
        test)
    print("Errors for the test data", "MSE:", mse,
          "-- RMSE:", rmse, "-- MAE:", mae)
    print('Precision:', precision, 'Recall:', recall, "Mapk:", mapk,
          'NDCG', ndcg)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    precisions.append(precision)
    recalls.append(recall)
    mapks.append(mapk)
    ndcgs.append(ndcg)
print('MEANS --', 'MAE:', np.mean(maes), 'MSE:', np.mean(mses),
      'RMSE:', np.mean(rmses))
print('MEANS --', 'Precision:', np.mean(precisions),
      'Recall:', np.mean(recalls),
      'MapK:', np.mean(mapks), 'NDCG', np.mean(ndcgs))
