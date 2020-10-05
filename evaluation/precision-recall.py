from collections import defaultdict

from surprise import Dataset
from surprise import SVD
from surprise.model_selection import KFold


def precision_recall_at_k(predictions, k=10, threshold= 3.5):
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
        number_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        