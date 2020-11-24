import pandas as pd


def read_comoda_kfold_splits(sets=5):
    train_test_sets = []
    for i in range(sets):
        train_set = pd.read_csv(
            'data/CoMoDa-kfold-split/train' + str(i) + '.csv',
            sep=';')
        test_set = pd.read_csv(
            'data/CoMoDa-kfold-split/test' + str(i) + '.csv',
            sep=';')
        train_test_sets.append((train_set, test_set))

    return train_test_sets
