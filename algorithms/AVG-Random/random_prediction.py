import pandas as pd
from math import sqrt
import random as random
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

maes = []
rmses = []

for _ in range(5):
    mae = 0
    mse = 0
    for _, row in data.iterrows():
        prediction = random.randint(1, 5)
        mae += abs(row['rating'] - prediction)
        mse += (row['rating'] - prediction)**2

    rmse = sqrt(mse/len(data.index))
    mae = mae/len(data.index)
    print('MAE', mae, 'RMSE', rmse)
    maes.append(mae)
    rmses.append(rmse)

print('Mean mae', np.mean(maes), 'Mean rmse', np.mean(rmses))
