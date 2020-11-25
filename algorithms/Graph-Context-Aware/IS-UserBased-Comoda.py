from itertools import product
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from collections import Counter
import time
from sklearn.metrics import mean_absolute_error
import argparse
# In case you cannot find the modules, include the following:
# import sys
# sys.path.insert(0, r'E:\git\Graph-Based-Recommender')
from algorithms.evaluation.mapk import calculate_map
from algorithms.evaluation.precision_recall import precision_recall_at_k
from dataloaders.read_comoda_kfold_splits import read_comoda_kfold_splits

parser = argparse.ArgumentParser(description='Itembased item splitting.')

parser.add_argument('--l', type=int,
                    help='Amount of jumps',
                    default=8)

parser.add_argument('--n', type=int,
                    help='Amount of neighbors',
                    default=10)

parser.add_argument('--k2', type=int,
                    help='Amount of items to consider',
                    default=10)

parser.add_argument('--threshold', type=int,
                    help='Minimum rating to be relevant',
                    default=3)

args = parser.parse_args()

train_test_set = read_comoda_kfold_splits()

rmses = []
maes = []
mapks = []
precisions = []
recalls = []

for train, test in train_test_set:
    topk_dict = {}
    relevant_dict = {}

    for _, relevant_item in test.iterrows():
        if relevant_item['rating'] >= args.threshold:
            if relevant_item['userID'] in relevant_dict.keys():
                relevant_dict[relevant_item['userID']].append(
                    relevant_item['itemID'])
            else:
                relevant_dict[relevant_item['userID']] = [
                    relevant_item['itemID']]

    # Step 0: Generate all values for Nc1, Nc2 ... Ncn
    items_distinct = train.itemID.unique()

    # Context dimensions
    dominantEmo_distinct = train.dominantEmo.unique()
    physical_distinct = train.physical.unique()
    context_dimensions_product = product(dominantEmo_distinct,
                                         physical_distinct)
    T_temp = list(product(context_dimensions_product, items_distinct))

    T = {k: v for k, v in enumerate(T_temp)}
    # Step 3: Generate the two-dimensional matrix of ratings
    # This is saved as a sparse matrix, since a lot of the ratings will
    # be unknown
    users_distinct = train.userID.unique()
    user_ids = {k: v for v, k in enumerate(users_distinct)}

    W = lil_matrix((len(users_distinct), len(T)), dtype=np.float32)

    # Iterate through ratings and put them in the matrix
    train_size = len(train)
    for index, row in train.iterrows():
        itemId = row['itemID']
        userID = row['userID']
        dominantEmo = row['dominantEmo']
        physical = row['physical']
        rating = row['rating']

        idx = list(T.keys())[list(T.values())
                             .index(((dominantEmo, physical), itemId))]

        W[user_ids[userID], idx] = rating

    W = csr_matrix(W)

    # Generate UZ_L based on the L value
    UZ2 = W.dot(np.transpose(W))
    UZL = UZ2
    # Use jumps to keep track of how many jumps we've made
    jumps = 2
    while jumps < 10:
        UZL = np.dot(UZ2, UZL)
        jumps += 2

    def predict(K1, K2, Ua, context, itemid=0):
        if Ua not in user_ids:
            print(f'User {Ua} not in trainset.')
            return [], 0
        fictional_id = user_ids[Ua]

        most_similar_users = []
        for le, ri in zip(UZL.indptr[:-1], UZL.indptr[1:]):
            n_row_pick = min(K1, ri - le)
            most_similar_users.append(
                UZL.indices[le + np.argpartition(
                    UZL.data[le:ri],
                    -n_row_pick)[-n_row_pick:]][0])

        # itemID, sum rating, rating count
        # k: itemID, v: (sum rating, rating count)
        rating_sum_counter = {}

        for sim_user_row in most_similar_users:
            for _, col in zip(*W[sim_user_row].nonzero()):
                if W[fictional_id, col]:
                    continue

                rating = W[sim_user_row, col]
                if col not in rating_sum_counter:
                    rating_sum_counter[col] = (rating, 1)
                else:
                    current_rating = rating_sum_counter[col]
                    new_rating = current_rating[0] + rating
                    new_count = current_rating[1] + 1
                    rating_sum_counter[col] = (new_rating, new_count)

        # Go through dictionary and sum values
        for k, v in rating_sum_counter.items():
            rating_sum_counter[k] = v[0]/v[1]

        # Sort list by highest values
        counted_recs = Counter(rating_sum_counter)

        filtered_results = []
        for item in counted_recs:
            if T[item][0] == context:
                filtered_results.append(T[item][1])

        filtered_results = filtered_results[:K2]

        idx = -1
        predicted_rating = 0
        for key, value in T.items():
            if value == (context, itemid):
                idx = key
                if key in rating_sum_counter:
                    predicted_rating = rating_sum_counter[idx]
                    return filtered_results, predicted_rating
                else:
                    break

        return filtered_results, predicted_rating

    test_set_size = len(test)

    start_time = time.time()

    actuals = []
    predictions = []
    rmse = 0
    mae = 0
    precision = 0
    precision_recall = []

    for index, user in test.iterrows():
        dominantEmo = user['dominantEmo']
        physical = user['physical']

        current_context = (dominantEmo, physical)

        item = user['itemID']
        user_id = user['userID']
        rating = user['rating']

        topK, prediction = predict(K1=args.n,
                                   K2=args.k2,
                                   Ua=user_id,
                                   context=current_context,
                                   itemid=item)

        topk_dict[user_id] = topK

        if prediction != 0:
            actuals.append(rating)
            predictions.append(prediction)
            precision_recall.append((user_id, item, rating, prediction, {}))

    mapk = calculate_map(topk_dict, relevant_dict, args.k2)
    mapks.append(mapk)

    if len(predictions) > 0:
        rmse = np.sqrt(np.mean((np.array(predictions) -
                                np.array(actuals))**2))
        mae = mean_absolute_error(actuals, predictions)
        rmses.append(rmse)
        maes.append(mae)
        precision, recall = precision_recall_at_k(
            precision_recall, args.k2, threshold=3)
        precisions.append(precision)
        recalls.append(recall)

print(f'RMSE: {(sum(rmses) / len(rmses)):.3f}')
print(f'MAE: {(sum(maes) / len(maes)):.3f}')
print(f'MAP@10: {(sum(mapks) / len(mapks)):.3f}')
print(f'Precision@10: {(sum(precisions) / len(precisions)):.3f}')
print(f'Recall@10: {(sum(recalls) / len(recalls)):.3f}')
