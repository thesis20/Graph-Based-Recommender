import math


def calculate_ndcg(top_k_dict, relevant_dict):
    """

    Parameters
    ----------
    top_k_dict: A dictionary of the top k results for each user in evaluation.
                Key should be ID. Value should be a list, comparable to the
                list used for relevant_dict.
    relevant_dict: A dictionary of all relevant items in the set for each user
                    in evaluation. The key should be the user ID, equal to
                    the keys in topk_dict.

    Returns the normalized discounted cumulative gain for all users, where
    the gain is a binary value of 1 if the item being recommended is relevant,
    and 0 if not.
    -------

    """
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

    return final_dcg / final_idcg
