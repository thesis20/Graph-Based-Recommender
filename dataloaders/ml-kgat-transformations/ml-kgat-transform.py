from dataloaders.movielens_data import read_ml_kfold_splits
import pandas as pd
from math import floor


knowledge = pd.read_csv('data/ml-100k-research/u.item',
                        sep='|',
                        encoding='latin',
                        names=['movieId', 'movie_title',
                               'release_date', 'video_release_date',
                               'IMDb_URL', 'unknown', 'Action',
                               'Adventure', 'Animation',
                               'Childrens', 'Comedy', 'Crime',
                               'Documentary', 'Drama', 'Fantasy',
                               'Film-Noir', 'Horror', 'Musical',
                               'Mystery', 'Romance', 'SciFi',
                               'Thriller', 'War', 'Western'])

data = read_ml_kfold_splits()

full_data = data[0][0].append(data[0][1])
full_data = pd.merge(full_data, knowledge, on='movieId')
user_list = ''
item_list = ''
user_dict = {}
item_dict = {}
for index, userId in enumerate(full_data['userId'].unique()):
    user_list += str(userId) + ' ' + str(index) + '\n'
    user_dict[userId] = index
for index, movieId in enumerate(full_data['movieId'].unique()):
    item_list += 'movie' + str(movieId) + ' ' + \
        str(index) + ' ' + str(index) + '\n'
    item_dict[movieId] = index

f = open("user_list.txt", "w")
f.write(user_list)
f.close()
f = open("item_list.txt", "w")
f.write(item_list)
f.close()

counter = 0
entity_list = ''
entity_dict = {}
for movieId in full_data['movieId'].unique():
    if movieId == '\\N':
        continue
    entity_list += 'movie' + str(movieId) + ' ' + str(counter) + '\n'
    entity_dict['movie'+str(movieId)] = counter
    counter += 1

genre_names = ['unknown', 'Action',
               'Adventure', 'Animation',
               'Childrens', 'Comedy', 'Crime',
               'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical',
               'Mystery', 'Romance', 'SciFi',
               'Thriller', 'War', 'Western']

for release_date in full_data['release_date'].unique():
    if not pd.isna(release_date):
        year = release_date[-4:]
        # round down the last 2 digits to nearest 10
        decade = floor(int(year) / 10) * 10

        if entity_dict.get('dec'+str(decade)) is None:
            entity_list += 'dec' + str(decade) + ' ' + str(counter) + '\n'
            entity_dict['dec'+str(decade)] = counter
            counter += 1

for genre in genre_names:
    entity_list += 'genre' + str(genre) + ' ' + str(counter) + '\n'
    entity_dict['genre'+str(genre)] = counter
    counter += 1

f = open("entity_list.txt", "w")
f.write(entity_list)
f.close()

# make knowledge graph file
kg_final = ''
for movieId in full_data['movieId'].unique():
    movie_entity_id = str(entity_dict['movie'+str(movieId)])
    knowledge_df_row = knowledge.loc[knowledge['movieId'] == movieId]

    for genre in genre_names:
        has_genre = knowledge_df_row[genre].values[0]
        if has_genre == 1:
            genre_entity_id = entity_dict['genre' + str(genre)]
            kg_final += movie_entity_id + ' ' + \
                str(0) + ' ' + str(genre_entity_id) + '\n'

    release_date = knowledge_df_row['release_date'].values[0]
    if not pd.isna(release_date):
        year = release_date[-4:]
        # round down the last 2 digits to nearest 10
        decade = floor(int(year) / 10) * 10
        decade_entity_id = entity_dict['dec'+str(decade)]
        kg_final += movie_entity_id + ' ' + \
            str(1) + ' ' + str(decade_entity_id) + '\n'


f = open("kg_final.txt", "w")
f.write(kg_final)
f.close()

fold_count = 0
for train, test in data:
    train_interaction_dict = {}
    train_output = ''
    for index, row in train.iterrows():
        userId = user_dict[row['userId']]
        movieId = item_dict.get(row['movieId'])
        if movieId is None:
            continue
        if train_interaction_dict.get(userId) is None:
            movie_set = set()
            train_interaction_dict[userId] = movie_set
            train_interaction_dict[userId].add(movieId)
        else:
            train_interaction_dict[userId].add(movieId)
    for userId, movieIds in train_interaction_dict.items():
        train_output += str(userId)
        for movieId in movieIds:
            train_output += ' ' + str(movieId)
        train_output += '\n'
    f = open("train" + str(fold_count) + ".txt", "w")
    f.write(train_output)
    f.close()

    test_interaction_dict = {}
    test_output = ''
    for index, row in test.iterrows():
        userId = user_dict[row['userId']]
        movieId = item_dict.get(row['movieId'])
        if movieId is None:
            continue
        if test_interaction_dict.get(userId) is None:
            movie_set = set()
            test_interaction_dict[userId] = movie_set
            test_interaction_dict[userId].add(movieId)
        else:
            test_interaction_dict[userId].add(movieId)
    for userId, movieIds in test_interaction_dict.items():
        test_output += str(userId)
        for movieId in movieIds:
            test_output += ' ' + str(movieId)
        test_output += '\n'
    f = open("test" + str(fold_count) + ".txt", "w")
    f.write(test_output)
    f.close()

    fold_count += 1
