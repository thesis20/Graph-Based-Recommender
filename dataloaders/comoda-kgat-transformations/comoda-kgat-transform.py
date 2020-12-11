from dataloaders.read_comoda_kfold_splits import read_comoda_kfold_splits
import pandas as pd


knowledge = pd.read_csv('data/comoda-knowledge/itemmetadata2_v2.csv',
                        sep=';')
data = read_comoda_kfold_splits()

full_data = data[0][0].append(data[0][1])
full_data = pd.merge(full_data, knowledge,
                     left_on='itemID', right_on='item_ID')
user_list = ''
item_list = ''
user_dict = {}
item_dict = {}
for index, userId in enumerate(full_data['userID'].unique()):
    user_list += str(userId) + ' ' + str(index) + '\n'
    user_dict[userId] = index
for index, movieId in enumerate(full_data['itemID'].unique()):
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
for movieId in full_data['itemID'].unique():
    if movieId == 'NULL':
        continue
    entity_list += 'movie' + str(movieId) + ' ' + str(counter) + '\n'
    entity_dict['movie'+str(movieId)] = counter
    counter += 1

for director in full_data['Director'].unique():
    if director == 'NULL':
        continue
    if entity_dict.get('dir'+director) is None:
        entity_list += 'dir' + \
            str(director) + ' ' + str(counter) + '\n'
        entity_dict['dir'+str(director)] = counter
        counter += 1

for country in full_data['Country'].unique():
    if country == 'NULL':
        continue
    if entity_dict.get('country'+str(country)) is None:
        entity_list += 'country' + str(country) + ' ' + str(counter) + '\n'
        entity_dict['country'+str(country)] = counter
        counter += 1

for language in full_data['Language'].unique():
    if language == 'NULL':
        continue
    if entity_dict.get('lang'+language) is None:
        entity_list += 'lang' + str(language) + ' ' + str(counter) + '\n'
        entity_dict['lang'+str(language)] = counter
        counter += 1

for year in full_data['Year'].unique():
    if year == 'NULL':
        continue
    if entity_dict.get('year'+str(year)) is None:
        entity_list += 'year' + str(year) + ' ' + str(counter) + '\n'
        entity_dict['year'+str(year)] = counter
        counter += 1

for genre in full_data['Genre1'].unique():
    if genre == 'NULL':
        continue
    if entity_dict.get('genre'+genre) is None:
        entity_list += 'genre' + str(genre) + ' ' + str(counter) + '\n'
        entity_dict['genre'+str(genre)] = counter
        counter += 1
for genre in full_data['Genre2'].unique():
    if genre == 'NULL':
        continue
    if entity_dict.get('genre'+genre) is None:
        entity_list += 'genre' + str(genre) + ' ' + str(counter) + '\n'
        entity_dict['genre'+str(genre)] = counter
        counter += 1
for genre in full_data['Genre3'].unique():
    if genre == 'NULL':
        continue
    if entity_dict.get('genre'+genre) is None:
        entity_list += 'genre' + str(genre) + ' ' + str(counter) + '\n'
        entity_dict['genre'+str(genre)] = counter
        counter += 1

for actor in full_data['Actor1'].unique():
    if actor == 'NULL':
        continue
    if entity_dict.get('actor'+actor) is None:
        entity_list += 'actor' + str(actor) + ' ' + str(counter) + '\n'
        entity_dict['actor'+str(actor)] = counter
        counter += 1
for actor in full_data['Actor2'].unique():
    if actor == 'NULL':
        continue
    if entity_dict.get('actor'+actor) is None:
        entity_list += 'actor' + str(actor) + ' ' + str(counter) + '\n'
        entity_dict['actor'+str(actor)] = counter
        counter += 1
for actor in full_data['Actor3'].unique():
    if genre == 'NULL':
        continue
    if entity_dict.get('actor'+actor) is None:
        entity_list += 'actor' + str(actor) + ' ' + str(counter) + '\n'
        entity_dict['actor'+str(actor)] = counter
        counter += 1

f = open("entity_list.txt", "w")
f.write(entity_list)
f.close()

# make knowledge graph file
kg_final = ''
for movieId in full_data['itemID'].unique():
    movie_entity_id = str(entity_dict['movie'+str(movieId)])
    knowledge_df_row = knowledge.loc[knowledge['item_ID'] == movieId]

    director = knowledge_df_row['Director'].values[0]
    if director != 'NULL':
        director_entity_id = entity_dict['dir' + director]
        kg_final += movie_entity_id + ' ' + \
            str(0) + ' ' + str(director_entity_id) + '\n'

    country = knowledge_df_row['Country'].values[0]
    if country != 'NULL':
        country_entity_id = entity_dict['country' + country]
        kg_final += movie_entity_id + ' ' + \
            str(1) + ' ' + str(country_entity_id) + '\n'

    language = knowledge_df_row['Language'].values[0]
    if language != 'NULL':
        language_entity_id = entity_dict['lang' + language]
        kg_final += movie_entity_id + ' ' + \
            str(2) + ' ' + str(language_entity_id) + '\n'

    year = knowledge_df_row['Year'].values[0]
    if year != 'NULL':
        year_entity_id = entity_dict['year' + str(year)]
        kg_final += movie_entity_id + ' ' + \
            str(3) + ' ' + str(year_entity_id) + '\n'

    genre = knowledge_df_row['Genre1'].values[0]
    if genre != 'NULL':
        genre_entity_id = entity_dict['genre' + genre]
        kg_final += movie_entity_id + ' ' + \
            str(4) + ' ' + str(genre_entity_id) + '\n'
    genre = knowledge_df_row['Genre2'].values[0]
    if genre != 'NULL':
        genre_entity_id = entity_dict['genre' + genre]
        kg_final += movie_entity_id + ' ' + \
            str(4) + ' ' + str(genre_entity_id) + '\n'
    genre = knowledge_df_row['Genre3'].values[0]
    if genre != 'NULL':
        genre_entity_id = entity_dict['genre' + genre]
        kg_final += movie_entity_id + ' ' + \
            str(4) + ' ' + str(genre_entity_id) + '\n'

    actor = knowledge_df_row['Actor1'].values[0]
    if actor != 'NULL':
        actor_entity_id = entity_dict['actor' + actor]
        kg_final += movie_entity_id + ' ' + \
            str(5) + ' ' + str(actor_entity_id) + '\n'
    actor = knowledge_df_row['Actor2'].values[0]
    if actor != 'NULL':
        actor_entity_id = entity_dict['actor' + actor]
        kg_final += movie_entity_id + ' ' + \
            str(5) + ' ' + str(actor_entity_id) + '\n'
    actor = knowledge_df_row['Actor3'].values[0]
    if actor != 'NULL':
        actor_entity_id = entity_dict['actor' + actor]
        kg_final += movie_entity_id + ' ' + \
            str(5) + ' ' + str(actor_entity_id) + '\n'


f = open("kg_final.txt", "w")
f.write(kg_final)
f.close()

fold_count = 0
for train, test in data:
    train_interaction_dict = {}
    train_output = ''
    for index, row in train.iterrows():
        userId = user_dict[row['userID']]
        movieId = item_dict.get(row['itemID'])
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
        userId = user_dict[row['userID']]
        movieId = item_dict.get(row['itemID'])
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
