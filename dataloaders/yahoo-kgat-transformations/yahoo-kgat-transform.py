from dataloaders.read_yahoo_kfold_splits import read_yahoo_kfold_splits
from dataloaders.yahoo_knowledge import read_kgat_knowledge
import pandas as pd
import re


knowledge = read_kgat_knowledge()
data = read_yahoo_kfold_splits()

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

for mpaa_rating in full_data['MPAA_rating'].unique():
    if mpaa_rating == '\\N':
        continue
    mpaa_rating_split = mpaa_rating.split(' ')
    if entity_dict.get('mpaa'+mpaa_rating_split[0]) is None:
        entity_list += 'mpaa' + \
            str(mpaa_rating_split[0]) + ' ' + str(counter) + '\n'
        entity_dict['mpaa'+str(mpaa_rating_split[0])] = counter
        counter += 1

for distributor in full_data['distributor'].unique():
    if distributor == '\\N':
        continue
    distributor = re.sub(r'[^a-zA-Z]+', '', distributor)
    if entity_dict.get('dist'+str(distributor)) is None:
        entity_list += 'dist' + str(distributor) + ' ' + str(counter) + '\n'
        entity_dict['dist'+str(distributor)] = counter
        counter += 1

for genres in full_data['genres'].unique():
    if genres == '\\N':
        continue
    genre_split = genres.split('|')
    for genre in genre_split:
        if entity_dict.get('genre'+genre) is None:
            entity_list += 'genre' + str(genre) + ' ' + str(counter) + '\n'
            entity_dict['genre'+str(genre)] = counter
            counter += 1

for directorIds in full_data['directorIds'].unique():
    if directorIds == '\\N':
        continue
    directorIds_split = directorIds.split('|')
    for dirId in directorIds_split:
        if entity_dict.get('dir'+dirId) is None:
            entity_list += 'dir' + str(dirId) + ' ' + str(counter) + '\n'
            entity_dict['dir'+str(dirId)] = counter
            counter += 1

for actorIds in full_data['actorIds'].unique():
    if actorIds == '\\N':
        continue
    actorIds_split = actorIds.split('|')
    for actorId in actorIds_split:
        if entity_dict.get('actor'+actorId) is None:
            entity_list += 'actor' + str(actorId) + ' ' + str(counter) + '\n'
            entity_dict['actor'+str(actorId)] = counter
            counter += 1

f = open("entity_list.txt", "w")
f.write(entity_list)
f.close()

# make knowledge graph file
kg_final = ''
for index, row in full_data.iterrows():
    movie_entity_id = str(entity_dict['movie'+str(row['movieId'])])
    mpaa_rating = row['MPAA_rating'].split(' ')[0]
    if mpaa_rating != '\\N':
        mpaa_entity_id = entity_dict['mpaa' + mpaa_rating]
        kg_final += movie_entity_id + ' ' + \
            str(0) + ' ' + str(mpaa_entity_id) + '\n'

    distributor = row['distributor']
    if distributor != '\\N':
        distributor = re.sub(r'[^a-zA-Z]+', '', distributor)
        distributor_entity_id = entity_dict['dist' + distributor]
        kg_final += movie_entity_id + ' ' + \
            str(1) + ' ' + str(distributor_entity_id) + '\n'

    genres = row['genres']
    if genres != '\\N':
        genre_split = genres.split('|')
        for genre in genre_split:
            genre_entity_id = entity_dict['genre' + genre]
            kg_final += movie_entity_id + ' ' + \
                str(2) + ' ' + str(genre_entity_id) + '\n'

    directorIds = row['directorIds']
    if directorIds != '\\N':
        directorIds_split = directorIds.split('|')
        for dirId in directorIds_split:
            dir_entity_id = entity_dict['dir' + dirId]
            kg_final += movie_entity_id + ' ' + \
                str(3) + ' ' + str(dir_entity_id) + '\n'

    actorIds = row['actorIds']
    if actorIds != '\\N':
        actorIds_split = actorIds.split('|')
        for actorId in actorIds_split:
            actor_entity_id = entity_dict['actor' + actorId]
            kg_final += movie_entity_id + ' ' + \
                str(4) + ' ' + str(actor_entity_id) + '\n'


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
