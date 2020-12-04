import pandas as pd


def read_comoda_kfold_splits(sets=5):
    train_test_sets = []
    for i in range(sets):
        train_set = pd.read_csv(
            'data/CoMoDa-kfold-split/' + str(i) + '/train.csv',
            sep=';')
        test_set = pd.read_csv(
            'data/CoMoDa-kfold-split/' + str(i) + '/test.csv',
            sep=';')
        train_test_sets.append((train_set, test_set))
    return train_test_sets
