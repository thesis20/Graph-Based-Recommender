from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split
from dataloaders import movielens_data as loader
from surprise import Reader


def built_in_100k():
    # Load the movielens-100k data set (download it if needed),
    data = Dataset.load_builtin('ml-100k')

    return data


def custom_pandas_100k():
    # Load movieLens-100k from pandas dataframe using our data loader
    movies, ratings = loader.load_data_ml100k()

    # Ratings data is reordered such that it fits Surprise - item, rating, user
    ratings_frame = ratings[['userId', 'movieId', 'rating']]
    
    # Reader is a Surpise class that is necessary for parsing files
    # For pandas dataframes, it simply requires a rating scale.
    # The scale in this case is 1-5.
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(ratings_frame, reader)

    return data


def svd_cross_validate():
    data = custom_pandas_100k()
    algo = SVD()

    # Run 5-fold cross-validation and print results
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


def svd_train_test_split():
    data = custom_pandas_100k()

    # Split data, training is 80% and test is 20%
    train_set, test_set = train_test_split(data, test_size=.20)
    algo = SVD()

    # Train on trainings et
    algo.fit(train_set)
    # Predict ratings for test set
    predictions = algo.test(test_set)

    # Compute RMSE
    accuracy.rmse(predictions)
