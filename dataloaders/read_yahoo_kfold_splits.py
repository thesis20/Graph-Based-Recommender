import pandas as pd


def read_yahoo_kfold_splits(folds=5):
    train_test_sets = []
    for i in range(folds):
        train_set = pd.read_csv(
            'data/yahoo-kfold-splits/' + str(i) + '/train.csv',
            sep=';')
        test_set = pd.read_csv(
            'data/yahoo-kfold-splits/' + str(i) + '/test.csv',
            sep=';', index_col=False)
        train_test_sets.append((train_set, test_set))

    return train_test_sets
