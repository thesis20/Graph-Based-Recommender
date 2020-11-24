import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold
import numpy as np
# movielens
data = pd.read_csv('../data/ml-100k-research/movielens_uirc.csv',
                   sep=';', names=['userId', 'movieId', 'rating',
                                   'timeOfDay', 'dayOfWeek'],
                   header=None, index_col=False)
# yahoo
# data = pd.read_csv('../data/yahoo-movies/yahoo-movies.csv', sep=',')

# CoMoDa
""" data = pd.read_csv('../data/CoMoDa.csv', sep=';')
# Prune users that have not rated more than 5 movies
data = data[data['userID'].isin(data['userID'].value_counts()[
    data['userID'].value_counts() > 5].index)] """

kf = KFold(5, shuffle=True)

maes = []
mses = []
rmses = []
for train, test in kf.split(data):
    train_set = data.iloc[train]
    test_set = data.iloc[test]

    average_item_ratings = {}
    for _, row in train_set.iterrows():
        average_item_ratings[row['movieId']] = train_set.loc[
            train_set['movieId'] == row['movieId']]['rating'].mean()

    mae = 0
    mse = 0
    hits = 0
    for _, row in test_set.iterrows():
        average_item_rating = average_item_ratings.get(row['movieId'])
        if average_item_rating is not None:
            hits += 1
            mae += abs(row['rating'] - average_item_rating)
            mse += (row['rating'] - average_item_rating)**2

    rmse = sqrt(mse/hits)
    mae = mae/hits
    print('MAE', mae, 'RMSE', rmse)
    maes.append(mae)
    rmses.append(rmse)

print('Mean mae', np.mean(maes), 'Mean rmse', np.mean(rmses))
