import pandas as pd
import random
import math


def random_list_predictor(all_items, k):
    return random.sample(all_items, k)


def calculate_map(topk_dict, relevant_dict, k_val):
    average_precisions = []
    number_of_users = len(topk_dict)

    for userId, items in topk_dict.items():
        score = 0.0
        hit_rate = 0.0

        counter = 1
        for itemId in items:
            if userId in relevant_dict:
                if itemId in relevant_dict[userId]:
                    hit_rate += 1
                    score += hit_rate / counter
            counter += 1

        if relevant_dict.get(userId) is None:
            average_precisions.append(0.0)
        elif len(relevant_dict[userId]) < k_val:
            average_precisions.append(score / len(relevant_dict[userId]))
        else:
            average_precisions.append(score / k_val)

    return (1 / number_of_users) * sum(average_precisions)


def calculate_ndcg(top_k_dict, relevant_dict):
    dcg, idcg, gain = 0.0, 0.0, 1
    number_of_users = len(top_k_dict)

    for key, value in top_k_dict.items():
        position = 0
        number_of_relevant_items = 0
        for val in value:
            position += 1
            if key in relevant_dict.keys():
                if val in relevant_dict[key]:
                    number_of_relevant_items += 1
                    dcg += gain / math.log2(position + 1)

        if number_of_relevant_items > 0:
            for i in range(number_of_relevant_items):
                idcg += 1 / math.log2((i+1) + 1)

    final_dcg = (1 / number_of_users) * dcg
    final_idcg = (1 / number_of_users) * idcg
    if final_idcg > 0:
        return final_dcg / final_idcg
    else:
        return 0


def precision_at_k(top_k_dict, relevant_dict, k):
    precisions = []

    for user, items in top_k_dict.items():
        num_rel = 0
        for item in items:
            if item in relevant_dict[user]:
                num_rel += 1
        precisions.append(num_rel / k)

    return sum(precisions) / len(precisions)


def recall_at_k(top_k_dict, relevant_dict, k):
    recall_dict = {}  # userId : num_rel
    for user, items in top_k_dict.items():
        num_rel = 0
        for item in items:
            if item in relevant_dict[user]:
                num_rel += 1
        recall_dict[user] = num_rel

    recalls = []
    for user, num_rel in recall_dict.items():
        recalls.append(num_rel / len(relevant_dict[user]))

    return sum(recalls) / len(recalls)


ndcgs = []
mapks = []
recalls = []
precisions = []
recalls = []


# This uses the same folds as NGCF and LightGCN, so an edge list for each user
# and their interactions.
# Assumes the data is in folders named dataset_name0, dataset_name1 etc.
dataset_name = 'frappe'
amount_of_folds = 5

for fold_no in range(amount_of_folds - 1):
    fold_no = str(fold_no)
    train = open(dataset_name + fold_no + '/train.txt', "r")
    test = open(dataset_name + fold_no + '/test.txt', "r")

    pd_items = pd.read_csv('frappe' + fold_no + '/item_list.txt', sep=" ")
    item_list = set(pd_items['remap_id'].unique())

    user_relevant_items = {}
    for line in test:
        line_split = str.split(line, ' ')
        user_id = int(line_split[0])
        user_relevant_items[user_id] = set()
        for item in line_split[1:]:
            user_relevant_items[user_id].add(int(item))

    top_k_recs = {}
    random.seed(42)
    for user in user_relevant_items.keys():
        top_k_recs[user] = random_list_predictor(item_list, 10)

    mapk = calculate_map(top_k_recs, user_relevant_items, 10)
    ndcg = calculate_ndcg(top_k_recs, user_relevant_items)
    precision = precision_at_k(top_k_recs, user_relevant_items, 10)
    recall = recall_at_k(top_k_recs, user_relevant_items, 10)
    mapks.append(mapk)
    ndcgs.append(ndcg)
    precisions.append(precision)
    recalls.append(recall)

mapk = sum(mapks) / len(mapks)
ndcg = sum(ndcgs) / len(ndcgs)
precision = sum(precisions) / len(precisions)
recall = sum(recalls) / len(recalls)
print(f'mapk: {mapk} - ndcg: {ndcg} - precision: {precision} \
    - recall: {recall}')
