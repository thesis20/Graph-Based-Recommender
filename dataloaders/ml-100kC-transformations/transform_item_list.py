import pandas as pd
from itertools import product

data = pd.read_csv('ml-100k/u1_context.csv', sep=';')

items_distinct = data.movieId.unique()
timeofday_distinct = data.timeofday.unique() # Included
dayofweek_distinct = data.dayofweek.unique() # Included

context_dimensions_product = product(timeofday_distinct, dayofweek_distinct)
T_temp = list(product(context_dimensions_product, items_distinct))
T = {k: v for k, v in enumerate(T_temp)}

output = ""
for i, v in T.items():
    output += (str(i) + " " + str(i) + " " + str(v[0][0])
               + str(v[0][1]) + str(v[1]) + "\n")


f = open("item_list.txt", "w")
f.write(output)
f.close()
