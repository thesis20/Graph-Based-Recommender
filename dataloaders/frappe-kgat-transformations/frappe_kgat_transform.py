from dataloaders.read_frappe_kfold_splits import read_frappe_kfold_splits
import pandas as pd


knowledge = pd.read_csv('data/Mobile_Frappe/frappe/meta.csv',
                        sep='\t', encoding='latin')
data = read_frappe_kfold_splits()

full_data = data[0][0].append(data[0][1])
full_data = pd.merge(full_data, knowledge,
                     on='item')
user_list = ''
item_list = ''
user_dict = {}
item_dict = {}
for index, userId in enumerate(full_data['user'].unique()):
    user_list += str(userId) + ' ' + str(index) + '\n'
    user_dict[userId] = index
for index, movieId in enumerate(full_data['item'].unique()):
    item_list += 'movie' + str(movieId) + ' ' + \
        str(index) + ' ' + str(index) + '\n'
    item_dict[movieId] = index

f = open("user_list.txt", "w")
f.write(str(user_list))
f.close()
f = open("item_list.txt", "w")
f.write(str(item_list))
f.close()

counter = 0
entity_list = ''
entity_dict = {}
for movieId in full_data['item'].unique():
    if movieId == 'unknown':
        continue
    entity_list += 'movie' + str(movieId) + ' ' + str(counter) + '\n'
    entity_dict['movie'+str(movieId)] = counter
    counter += 1

for category in full_data['category'].unique():
    if category == 'unknown':
        continue
    if entity_dict.get('category'+category) is None:
        entity_list += 'category' + \
            str(category) + ' ' + str(counter) + '\n'
        entity_dict['category'+str(category)] = counter
        counter += 1

for download in full_data['downloads'].unique():
    if download == 'unknown':
        continue
    if entity_dict.get('download'+str(download)) is None:
        entity_list += 'download' + str(download) + ' ' + str(counter) + '\n'
        entity_dict['download'+str(download)] = counter
        counter += 1

for developer in full_data['developer'].unique():
    if developer == 'unknown':
        continue
    if entity_dict.get('developer'+developer) is None:
        entity_list += 'developer' + str(developer) + ' ' + str(counter) + '\n'
        entity_dict['developer'+str(developer)] = counter
        counter += 1

for language in full_data['language'].unique():
    if language == 'unknown':
        continue
    if entity_dict.get('language'+str(language)) is None:
        entity_list += 'language' + str(language) + ' ' + str(counter) + '\n'
        entity_dict['language'+str(language)] = counter
        counter += 1

f = open("entity_list.txt", "w", encoding="utf-8")
f.write(str(entity_list))
f.close()

# make knowledge graph file
kg_final = ''
for movieId in full_data['item'].unique():
    movie_entity_id = str(entity_dict['movie'+str(movieId)])
    knowledge_df_row = knowledge.loc[knowledge['item'] == movieId]

    category = knowledge_df_row['category'].values[0]
    if category != 'unknown':
        category_entity_id = entity_dict['category' + category]
        kg_final += movie_entity_id + ' ' + \
            str(0) + ' ' + str(category_entity_id) + '\n'

    download = knowledge_df_row['downloads'].values[0]
    if download != 'unknown':
        download_entity_id = entity_dict['download' + download]
        kg_final += movie_entity_id + ' ' + \
            str(1) + ' ' + str(download_entity_id) + '\n'

    developer = knowledge_df_row['developer'].values[0]
    if developer != 'unknown':
        developer_entity_id = entity_dict['developer' + str(developer)]
        kg_final += movie_entity_id + ' ' + \
            str(2) + ' ' + str(developer_entity_id) + '\n'

    language = knowledge_df_row['language'].values[0]
    if language != 'unknown':
        language_entity_id = entity_dict['language' + language]
        kg_final += movie_entity_id + ' ' + \
            str(3) + ' ' + str(language_entity_id) + '\n'

f = open("kg_final.txt", "w", encoding='utf-8')
f.write(str(kg_final))
f.close()

fold_count = 0
for train, test in data:
    train_interaction_dict = {}
    train_output = ''
    for index, row in train.iterrows():
        userId = user_dict[row['user']]
        movieId = item_dict.get(row['item'])
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
        userId = user_dict[row['user']]
        movieId = item_dict.get(row['item'])
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
