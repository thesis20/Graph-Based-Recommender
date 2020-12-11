from collections import defaultdict

_folds = 5
_relevant_threshold = 3


def write_user_list(user_ids):
    '''

    Parameters
    ----------
    user_ids is a dict containing original id as key.
    This key is printed as the original id, and remap id is based on index from
    0 to x
    -------

    '''
    with open('user_list.txt', 'w') as user_list_file:
        user_list_file.write('org_id remap_id\n')
        for index, key in enumerate(user_ids):
            user_list_file.write(key + ' ' + str(index) + '\n')


def write_item_list(item_ids):
    '''

    Parameters
    ----------
    item_ids is a dict containing original item id as key
    Printed as original id and remapping is based on index
    -------

    '''
    with open('item_list.txt', 'w') as item_list_file:
        item_list_file.write('org_id remap_id\n')
        for index, key in enumerate(item_ids):
            item_list_file.write(key + ' ' + str(index) + '\n')


def write_test_split(user_items_dict, i):
    '''

    Parameters
    ----------
    user_items_dict contains a dictionary with a user id as key, and a string
    of item ids separated by space as value
    i is the current fold
    -------

    '''
    with open('test' + str(i) + '.txt', 'w') as test_writer:
        for key, value in user_items_dict.items():
            test_writer.write(str(key) + ' ' + str(value) + '\n')


def write_train_split(user_items_dict, i):
    with open('train' + str(i) + '.txt', 'w') as train_writer:
        for key, value in user_items_dict.items():
            train_writer.write(str(key) + ' ' + str(value) + '\n')


def update_user_items_dict(user_items_dict, remap_user_id, remap_item_id):
    '''

    Parameters
    ----------
    user_items_dict contains a dictionary with a user id as key, and a string
    of item ids separated by space as value
    remap_user_id is a remapped user id
    remap_item_id is a remapped item id

    Returns a dictionary updated by adding the item id to the string for the
    user or adding the user to the dictionary as well as the item as the value
    for that user
    -------

    '''
    if remap_user_id in user_items_dict.keys():
        user_items_dict[remap_user_id] = \
            user_items_dict[remap_user_id] + " " \
            + remap_item_id
    else:
        user_items_dict[remap_user_id] = remap_item_id

    return user_items_dict


def read_and_write_split(path, user_remapping, item_remapping,
                         positive_split=False, true_rating_col=0,
                         positive_threshold=3, prune=False):
    '''

    Parameters
    ----------
    path is the path for the data
    user_remapping is a dict of the original id as key and a counter of how
    many users has been found in the dataset
    item_remapping is the same as user_emapping, but for items
    positive_split is a bool to control whether or not we look at only data
    that exceeds a certain value
    true_rating_col is the column in the data in which the rating is observed
    positive_threshold is the threshold for when a rating is defined as
    positive
    prune is a bool controlling whether or not the data should be pruned

    Returns
    -------

    '''
    global _folds
    for i in range(_folds):
        user_items_train_dict = defaultdict()
        user_items_test_dict = defaultdict()
        if prune:
            user_items_train_dict_pruned = defaultdict()
            user_items_test_dict_pruned = defaultdict()

        with open(path + '/' + str(i) + '/train.csv') as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.split(';')
                    user_id = line[0]
                    item_id = line[1]

                    remap_user_id = str(user_remapping.get(user_id))
                    remap_item_id = str(item_remapping.get(item_id))

                    if positive_split:
                        if int(line[true_rating_col]) > positive_threshold:
                            user_items_train_dict = update_user_items_dict(
                                user_items_train_dict, remap_user_id,
                                remap_item_id)
                    else:
                        user_items_train_dict = update_user_items_dict(
                            user_items_train_dict, remap_user_id,
                            remap_item_id)

        with open(path + '/' + str(i) + '/test.csv') as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.split(';')
                    user_id = line[0]
                    item_id = line[1]

                    remap_user_id = str(user_remapping.get(user_id))
                    remap_item_id = str(item_remapping.get(item_id))

                    if positive_split:
                        if int(line[true_rating_col]) > positive_threshold:
                            user_items_test_dict = update_user_items_dict(
                                user_items_test_dict, remap_user_id,
                                remap_item_id)
                    else:
                        user_items_test_dict = update_user_items_dict(
                            user_items_test_dict, remap_user_id,
                            remap_item_id)

        if prune:
            for key, value in user_items_train_dict.items():
                ratings = value.split(' ')
                if len(ratings) >= 5:
                    user_items_train_dict_pruned[key] = \
                        user_items_train_dict[key]

            for key, value in user_items_test_dict.items():
                ratings = value.split(' ')
                if len(ratings) >= 5:
                    user_items_test_dict_pruned[key] = \
                        user_items_test_dict[key]

            write_train_split(user_items_train_dict_pruned, i)
            write_test_split(user_items_test_dict_pruned, i)
        else:
            write_train_split(user_items_train_dict, i)
            write_test_split(user_items_test_dict, i)


def convert_data_yahoo(full_data_path, k_fold_path):
    '''

    Parameters
    ----------
    full_data_path is the path to the file of the full data, such that users
    and items can be remapped
    k_fold_path is the path to the split data
    -------

    '''
    global _folds
    item_remapping, user_remapping = {}, {}
    item_counter, user_counter = 0, 0

    with open(full_data_path) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.split(',')
                user_id = line[1]
                item_id = line[2]

                if item_id not in item_remapping.keys():
                    item_remapping[item_id] = item_counter
                    item_counter += 1
                if user_id not in user_remapping.keys():
                    user_remapping[user_id] = user_counter
                    user_counter += 1

    for i in range(_folds):
        read_and_write_split(k_fold_path, user_remapping, item_remapping)

    write_user_list(user_remapping)
    write_item_list(item_remapping)


def convert_data_comoda(full_data_path, k_fold_path):
    '''

    Parameters
    ----------
    full_data_path is the path to the file of the full data, such that users
    and items can be remapped
    k_fold_path is the path to the split data
    -------

    '''
    global _folds
    item_remapping, user_remapping = {}, {}
    item_counter, user_counter = 0, 0

    with open(full_data_path) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip().split(';')
                user_id = line[0]
                item_id = line[1]

                if item_id not in item_remapping.keys():
                    item_remapping[item_id] = item_counter
                    item_counter += 1
                if user_id not in user_remapping.keys():
                    user_remapping[user_id] = user_counter
                    user_counter += 1

    for i in range(_folds):
        read_and_write_split(k_fold_path, user_remapping, item_remapping,
                             prune=True)

    write_user_list(user_remapping)
    write_item_list(item_remapping)


def convert_data_movielens(full_data_path, k_fold_path):
    '''

    Parameters
    ----------
    full_data_path is the path to the file of the full data, such that users
    and items can be remapped
    k_fold_path is the path to the split data
    -------

    '''
    global _folds
    item_remapping, user_remapping = {}, {}
    item_counter, user_counter = 0, 0

    with open(full_data_path) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.split()
                user_id = line[0]
                item_id = line[1]

                if item_id not in item_remapping.keys():
                    item_remapping[item_id] = item_counter
                    item_counter += 1
                if user_id not in user_remapping.keys():
                    user_remapping[user_id] = user_counter
                    user_counter += 1

    for i in range(_folds):
        read_and_write_split(k_fold_path, user_remapping, item_remapping)

    write_user_list(user_remapping)
    write_item_list(item_remapping)


def convert_data_yahoo_positive(full_data_path, k_fold_path):
    '''

    Parameters
    ----------
    full_data_path is the path to the file of the full data, such that users
    and items can be remapped
    k_fold_path is the path to the split data
    -------

    '''
    global _folds
    global _relevant_threshold
    item_remapping, user_remapping = {}, {}
    item_counter, user_counter = 0, 0

    with open(full_data_path) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.split(',')
                if int(line[4]) >= _relevant_threshold:
                    user_id = line[1]
                    item_id = line[2]

                    if item_id not in item_remapping.keys():
                        item_remapping[item_id] = user_counter
                        user_counter += 1
                    if user_id not in user_remapping.keys():
                        user_remapping[user_id] = item_counter
                        item_counter += 1

        for i in range(_folds):
            read_and_write_split(k_fold_path, user_remapping, item_remapping,
                                 positive_split=True, true_rating_col=3,
                                 positive_threshold=_relevant_threshold)

    write_user_list(user_remapping)
    write_item_list(item_remapping)


def convert_data_frappe(full_data_path, k_fold_path):
    '''

    Parameters
    ----------
    full_data_path is the path to the file of the full data, such that users
    and items can be remapped
    k_fold_path is the path to the split data
    -------

    '''
    global _folds
    item_remapping, user_remapping = {}, {}
    item_counter, user_counter = 0, 0

    with open(full_data_path) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.split()
                user_id = line[0]
                item_id = line[1]

                if item_id not in item_remapping.keys():
                    item_remapping[item_id] = user_counter
                    user_counter += 1
                if user_id not in user_remapping.keys():
                    user_remapping[user_id] = item_counter
                    item_counter += 1

    for i in range(_folds):
        read_and_write_split(k_fold_path, user_remapping, item_remapping)

    write_user_list(user_remapping)
    write_item_list(item_remapping)
