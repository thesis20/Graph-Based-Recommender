import pandas as pd


def read_frappe_kfold_splits(folds=5):
    train_test_sets = []
    for i in range(folds):
        train_set = pd.read_csv(
            'data/frappe-kfold/' + str(i) + '/train.csv',
            sep=';')
        test_set = pd.read_csv(
            'data/frappe-kfold/' + str(i) + '/test.csv',
            sep=';')
        train_test_sets.append((train_set, test_set))
    return train_test_sets
