import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold
import numpy as np

# movielens
""" data = pd.read_csv('../data/ml-100k-research/movielens_uirc.csv',
                   sep=';', names=['userId', 'movieId', 'rating',
                                   'timeOfDay', 'dayOfWeek'],
                   header=None, index_col=False) """
                   
# yahoo
# data = pd.read_csv('../data/yahoo-movies/yahoo-movies.csv', sep=',')

# CoMoDa
data = pd.read_csv('../data/CoMoDa.csv', sep=';')
# Prune users that have not rated more than 5 movies
data = data[data['userID'].isin(data['userID'].value_counts()[
    data['userID'].value_counts() > 5].index)]

kf = KFold(5, shuffle=True)

maes = []
mses = []
rmses = []
for train, test in kf.split(data):
    train_set = data.iloc[train]
    test_set = data.iloc[test]

    average_user_ratings = {}
    for _, row in train_set.iterrows():
        average_user_ratings[row['userID']] = train_set.loc[
            train_set['userID'] == row['userID']]['rating'].mean()

    mae = 0
    mse = 0
    hits = 0
    for _, row in test_set.iterrows():
        average_user_rating = average_user_ratings.get(row['userID'])
        if average_user_rating is not None:
            hits += 1
            mae += abs(row['rating'] - average_user_rating)
            mse += (row['rating'] - average_user_rating)**2

    rmse = sqrt(mse/hits)
    mae = mae/hits
    print('MAE', mae, 'RMSE', rmse)
    maes.append(mae)
    rmses.append(rmse)

print('Mean mae', np.mean(maes), 'Mean rmse', np.mean(rmses))
