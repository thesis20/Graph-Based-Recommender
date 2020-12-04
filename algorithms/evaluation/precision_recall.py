from collections import defaultdict


def precision_recall_at_k(predictions, k=10, threshold=3):
    """Precision and recall at k for each user"""

    user_estim_true = defaultdict(list)
    for user_id, _, true_rating, estimated, _ in predictions:
        user_estim_true[user_id].append((estimated, true_rating))

    precisions = dict()
    recalls = dict()

    for user_id, user_ratings in user_estim_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        number_rel = sum((true_rating >= threshold) for (_, true_rating)
                         in user_ratings)
        number_rel_rec_k = sum(((true_rating >= threshold) and
                                (estim_rating >= threshold)) for
                               (estim_rating, true_rating) in user_ratings[:k])
        precisions[user_id] = number_rel_rec_k / k
        recalls[user_id] = number_rel_rec_k / number_rel \
            if number_rel != 0 else 0

    average_precision = sum(precisions.values()) / len(precisions)

    average_recall = sum(recalls.values()) / len(recalls)

    return average_precision, average_recall
