def calculate_map(topk_dict, relevant_dict, k_val):
    """

    Parameters
    ----------
    topk_dict: A dictionary of the top k results for each user in evaluation.
                Key should be ID. Value should be a list, comparable to the
                list used for relevant_dict.
    relevant_dict: A dictionary of all relevant items in the set for each user
                    in evaluation. The key should be the user ID, equal to
                    the keys in topk_dict.
    k_val: The number of items in each list of recommendations

    Returns the average of the sum of the averaged precisions for each user
    -------

    """
    average_precisions = []
    number_of_users = len(topk_dict)
    score = 0.0
    hit_rate = 0.0

    for i in number_of_users:
        for index, value in enumerate(topk_dict[i], start=1):
            if value in relevant_dict[i]:
                hit_rate += 1
                score += hit_rate / index

        if not relevant_dict:
            average_precisions.append(0.0)
        elif len(relevant_dict) < k_val:
            average_precisions.append(score / len(relevant_dict))
        else:
            average_precisions.append(score / k_val)

    return (1 / number_of_users) * sum(average_precisions)
