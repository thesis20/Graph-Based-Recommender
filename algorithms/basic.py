"""Basic modules for graph based recsys."""
import pandas as pd
import networkx as nx
import math
import numpy as np

# Load data
dt_dir_name = '../data/ml-100k'

rdata = pd.read_csv(dt_dir_name + '/' + 'ratings.csv')
tagdata = pd.read_csv(dt_dir_name + '/' + 'tags.csv')
movies = pd.read_csv(dt_dir_name + '/' + 'movies.csv')
rdata.shape

rdata = pd.merge(movies, rdata, on='movieId')

rdata['userId'] = 'u' + rdata['userId'].astype(str)
rdata['movieId'] = 'm' + rdata['movieId'].astype(str)


# Generate graph
# First we add the nodes
G = nx.Graph()
G.add_nodes_from(rdata.userId, bipartite=0)
G.add_nodes_from(rdata.movieId, bipartite=1)

# Then the weight for edges
G.add_weighted_edges_from([uId, mId, rating] for (uId, mId, rating)
                          in rdata[['userId', 'movieId', 'rating']].to_numpy())


# Visualizing the graph
# =============================================================================
# from networkx import *
# import matplotlib.pyplot as plt
# color_map = []
# for node in G.nodes:
#     if str(node).startswith('u'):
#         color_map.append('yellow')
#     else:
#         color_map.append('green')
# pos = nx.spring_layout(G)
# plt.figure(3, figsize=(12, 12))
# nx.draw(G, pos, node_color=color_map)
# plt.show()
# =============================================================================


def get_recommendation(movieId):
    """Give a recommendation based on kNN."""
    neighbors_dict = {}
    for e in G.neighbors(movieId):
        for e2 in G.neighbors(e):
            if e == movieId:
                continue
            if str(e2).startswith('m'):
                neighbors = neighbors_dict.get(e2)
                if neighbors is None:
                    neighbors_dict.update({e2: [e]})
                else:
                    neighbors.append(e)
                    neighbors_dict.update({e2: neighbors})
    movies = []
    weight = []
    for key, values in neighbors_dict.items():
        w = 0.0
        for e in values:
            w = w+1/math.log(G.degree(e))
        movies.append(get_movie_name(key))
        weight.append(w)

    result = pd.Series(data=np.array(weight), index=movies)
    result.sort_values(inplace=True, ascending=False)
    return result


def get_movie_name(movieId):
    """Find the movie name based on id."""
    return movies.loc[movies['movieId'] == int(movieId[1:])]['title'].iloc[0]


def get_recommendations_sp(movieId):
    """Find closest elements with nx."""
    closest = nx.single_source_dijkstra_path_length(G, movieId,
                                                    weight='weight')
    movie_dict = {k: v for k, v in closest.items() if k.startswith('m')}
    return sorted(movie_dict.items(), key=lambda x: x[1])


isolates = nx.isolates(G)
for iso in isolates:
    print(iso)

chosen_movie = 'm1'
result = get_recommendations_sp(chosen_movie)
print('Recommendations for {}:\n'.format(get_movie_name(chosen_movie)))
for (key, value) in result[1:8]:
    print(get_movie_name(key), " :: ", value)
