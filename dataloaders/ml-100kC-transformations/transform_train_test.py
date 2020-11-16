import pandas as pd
from itertools import product

data = pd.read_csv('ml-100k/u1_context_test.csv', sep=';')

items_distinct = data.movieId.unique()
timeofday_distinct = data.timeofday.unique()
dayofweek_distinct = data.dayofweek.unique()

context_dimensions_product = product(timeofday_distinct, dayofweek_distinct)
T_temp = list(product(context_dimensions_product, items_distinct))
T = {k: v for k, v in enumerate(T_temp)}


user_interactions = dict()

i = 0
for _, row in data.iterrows():
    if i % 100 == 0:
        print(f"{i} of {len(data.index)}")
    fictive_movie_id = list(T.keys())[list(T.values())
                                      .index((
                                          (row['timeofday'], row['dayofweek']),
                                          row['movieId']))]
    if row['rating'] >= 3:
        if row['userId'] in user_interactions:
            user_interactions[row['userId']].append(fictive_movie_id)
        else:
            user_interactions[row['userId']] = [fictive_movie_id]
    i += 1

output = ""

for key, values in user_interactions.items():
    output += str(key)
    for value in values:
        output += " " + str(value)
    output += "\n"
    i += 1


f = open("test_1.txt", "w")
f.write(output)
f.close()
