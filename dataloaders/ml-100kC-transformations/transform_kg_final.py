import pandas as pd
from itertools import product

data = pd.read_csv('ml-100k/u1_context_test.csv', sep=';')

items_distinct = data.movieId.unique()
timeofday_distinct = data.timeofday.unique()
dayofweek_distinct = data.dayofweek.unique()

context_dimensions_product = product(timeofday_distinct, dayofweek_distinct)
T_temp = list(product(context_dimensions_product, items_distinct))
T = {k: v for k, v in enumerate(T_temp)}

items = pd.read_csv('ml-100k/u.item', sep='|',
                    names=['id', 'title', 'release', 'videorelease', 'imdb',
                           'unknown', 'action', 'adventure', 'animation',
                           'childrens', 'comedy', 'crime', 'documentary',
                           'drama', 'fantasy',  'film-noir', 'horror',
                           'musical', 'mystery', 'romance', 'scifi',
                           'thriller', 'war', 'western'], encoding='latin-1')
item_list = pd.read_csv('done/item_list.txt', sep=' ',
                        names=['orig', 'remap', 'free'])
entity_list = pd.read_csv('done/entity_list.txt', sep=' ',
                          names=['orig_id', 'remap_id'])

output = ""
i = 0
for row, value in items.iterrows():
    print(i)
    i += 1
    # For each fictional item derived from original item
    for k, v in T.items():
        if v[1] == value['id']:
            remap_id = k
            if value['unknown'] == 1:
                output += str(remap_id) + " 0 " + str(16500) + "\n"
            if value['action'] == 1:
                output += str(remap_id) + " 0 " + str(16501) + "\n"
            if value['adventure'] == 1:
                output += str(remap_id) + " 0 " + str(16502) + "\n"
            if value['animation'] == 1:
                output += str(remap_id) + " 0 " + str(16503) + "\n"
            if value['childrens'] == 1:
                output += str(remap_id) + " 0 " + str(16504) + "\n"
            if value['comedy'] == 1:
                output += str(remap_id) + " 0 " + str(16505) + "\n"
            if value['crime'] == 1:
                output += str(remap_id) + " 0 " + str(16506) + "\n"
            if value['documentary'] == 1:
                output += str(remap_id) + " 0 " + str(16507) + "\n"
            if value['drama'] == 1:
                output += str(remap_id) + " 0 " + str(16508) + "\n"
            if value['fantasy'] == 1:
                output += str(remap_id) + " 0 " + str(16509) + "\n"
            if value['film-noir'] == 1:
                output += str(remap_id) + " 0 " + str(16510) + "\n"
            if value['horror'] == 1:
                output += str(remap_id) + " 0 " + str(16511) + "\n"
            if value['musical'] == 1:
                output += str(remap_id) + " 0 " + str(16512) + "\n"
            if value['mystery'] == 1:
                output += str(remap_id) + " 0 " + str(16513) + "\n"
            if value['romance'] == 1:
                output += str(remap_id) + " 0 " + str(16514) + "\n"
            if value['scifi'] == 1:
                output += str(remap_id) + " 0 " + str(16515) + "\n"
            if value['thriller'] == 1:
                output += str(remap_id) + " 0 " + str(16516) + "\n"
            if value['war'] == 1:
                output += str(remap_id) + " 0 " + str(16517) + "\n"
            if value['western'] == 1:
                output += str(remap_id) + " 0 " + str(16518) + "\n"

f = open("kg_final_1.txt", "w")
f.write(output)
f.close()
