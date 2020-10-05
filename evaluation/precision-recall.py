from collections import defaultdict


def precision_recall_at_k(predictions, k=10, threshold= 4):
    """Precision and recall at k for each user"""

    user_estim_true = defaultdict(list)
    # Dictionary of users along with a tuple defining estimated and true rating
    for user_id, _, true_rating, estimated, _ in predictions:
        user_estim_true[user_id].append((estimated, true_rating))

    precisions = dict()
    recalls = dict()

    # Dictionary items() returns the list with all dictionary keys and values
    for user_id, user_ratings in user_estim_true.items():

        # Sort by the estimated rating from the tuple
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Get number of relevant items
        number_rel = sum((true_rating >= threshold) for (_, true_rating)
                         in user_ratings)

        # Get number of recommended items in top k, uses sequence slicing
        # to iterate through the k first elements which are the recommendations
        number_rec_k = sum((estim_rating >= threshold) for (estim_rating, _)
                           in user_ratings[:k])

        # Get number of relevant and recommended items
        number_rel_rec_k = sum(((true_rating >= threshold) and
                                (estim_rating >= threshold)) for
                               (estim_rating, true_rating) in user_ratings[:k])

        # Precision is the proportion of recommended items that are relevant
        # Add precision for each user to precision dictionary
        # When number of recommended items in top k is 0, it is undefined
        precisions[user_id] = number_rel_rec_k / number_rec_k \
            if number_rec_k != 0 else 0

        # Recall is the proportion of relevant items that are recommended
        # Add to dictionary
        recalls[user_id] = number_rel_rec_k / number_rel \
            if number_rel != 0 else 0

    average_precision = sum(precisions.values()) / len(precisions)

    average_recall = sum(recalls.values()) / len(recalls)

    return average_precision, average_recall
