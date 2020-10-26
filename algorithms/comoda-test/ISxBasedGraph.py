import pandas as pd
from itertools import product
import numpy as np
import networkx as nx
from operator import itemgetter


def load_data():
    """
    Load the ldos-comoda data and transform it"""

    data = pd.read_csv('../../data/CoMoDa/dataset.csv', sep=';')
    movie_names = pd.read_csv('../../data/CoMoDa/itemsTitles.csv', sep=';',
                              names=["movieId", "title"])
    return data, movie_names


def item_splitting(data):
    users_distinct = data.userID.unique()

    # Find all unique context dimensions
    items_distinct = data.itemID.unique()
    endEmo_distinct = data.endEmo.unique()
    dominantEmo_distinct = data.dominantEmo.unique()

    # Calculate the cartesian product of all context dimensions.
    # We currently have to limit the amount of context due to my poor computer
    # not having a death wish.
    Nc_temp = list(product(items_distinct, dominantEmo_distinct,
                           endEmo_distinct))

    Nc = {k: v for v, k in enumerate(Nc_temp)}

    # Create a lookup dictionary to quickly find the matrix entry for a user
    # Example: users_ids_dict[userId] = [index in users_distinct]
    user_ids = {k: v for v, k in enumerate(users_distinct)}

    # T is the itemset of fictional items
    users_by_T_matrix = np.zeros((len(users_distinct),
                                  len(list(Nc_temp))),
                                 dtype=np.dtype('float64'))

    for item in data.itertuples():
        # Alias entities
        userId = item[1]
        itemId = item[2]
        rating = item[3]
        endEmo = item[14]
        domEmo = item[15]

        # index for the item we'll find in users x T matrix
        idx = Nc[(itemId, domEmo, endEmo)]

        # Some users have rated the same movie twice in different context
        # We'll add the new rating and take the average of the two.
        if users_by_T_matrix[user_ids[userId]][idx] > 0.0:
            users_by_T_matrix[user_ids[userId]][idx] = users_by_T_matrix[
                user_ids[userId]][idx] + rating / 2
        else:
            users_by_T_matrix[user_ids[userId]][idx] = rating

    return users_by_T_matrix, Nc, users_distinct


def generate_graph(edge_list):
    edgelist = []
    users = []

    for index_r, row in enumerate(edge_list):
        users.append("u" + str(index_r))
        for index_i, item in enumerate(row):
            if item > 0:
                edgelist.append("u" + str(index_r) + " " + "i"
                                + str(index_i) + " " + str(item))

    G = nx.parse_edgelist(edgelist, data=(('weight', float), ))

    # fx = nx.draw_networkx(G, pos=nx.bipartite_layout(G, users))
    # plt.savefig('yes.png')

    return G, users


def calculate_similarity(graph, users, path_len=6):
    L = 2
    W = nx.algorithms.bipartite.matrix.biadjacency_matrix(
        graph, users, weight='weight').todense()

    Wt = np.transpose(W)
    UZ2 = np.dot(W, Wt)
    UZ = UZ2

    while L < path_len:
        UZ = np.dot(UZ2, UZ)
        L += 2

    return UZ


def get_rec(similarity, user_id, k, split_items, context):
    G = nx.Graph(similarity)
    distances = nx.single_source_dijkstra_path_length(G, source=user_id,
                                                      weight='weight')

    most_similar_users = list(dict(sorted(distances.items(), key=itemgetter(1),
                                          reverse=True)))

    # Find user row by their ID based on original sorting
    relative_user_id = find_user_index(user_id)

    recs = {}
    item_split_row = split_items[relative_user_id]
    not_rated_indices = np.argwhere(item_split_row == 0.0)

    for not_rated_item in not_rated_indices:
        summation = 0
        counter = 0
        for sim_user in most_similar_users:
            if split_items[sim_user][not_rated_item[0]] > 0:
                summation += split_items[sim_user][not_rated_item[0]]
                counter += 1

        if counter == 0:
            continue
        else:
            # Modulo the amount of items, currently hard-coded for testing
            recs[not_rated_item[0]] = (split_items[not_rated_item[0] % 121],
                                       summation/counter)

    res = [(k, v) for k, v in recs.items()]

    # Sort results so highest weight is first
    res1 = sorted(res, key=lambda s: s[1][1], reverse=True)

    # Filter results that doesn't fit the context
    filtered_results = [con for con in res1 if find_context(con, context)]

    return filtered_results[:k]


def find_context(item_id, context):
    item_id = item_id[0]
    for x in movie_lookup.items():
        if x[1] in movie_lookup.values():
            if context[0] == x[0][1] and context[1] == x[0][2]:
                return True
        return False


def find_user_index(user_id):
    return user_id-1
    # return index from user_list based on user_id


def get_title(item_id):
    for x, y in movie_lookup.items():
        if y == item_id:
            return movies.loc[movies['movieId'] == int(x[0])]['title'].iloc[0]


data, movies = load_data()
split_items, movie_lookup, user_list = item_splitting(data)
graph, users = generate_graph(split_items)
similarity = calculate_similarity(graph, users, 6)


# User ID you want to recommend for
USER_ID = 2

# Amount of recommendations you want
K = 10

# domEmo, endEmo
CONTEXT = (3, 3)

res = get_rec(similarity, USER_ID, K, split_items, CONTEXT)

print(f'Top {K} recommendations for user {USER_ID} in context {CONTEXT}:')

for item in res:
    print(get_title(item[0]))
