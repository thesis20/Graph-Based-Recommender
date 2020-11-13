import pandas as pd

data = pd.read_csv('ml-100k/u1_context.csv', sep=';')

user_ids = data['userId'].unique()

output = ""
for index, id in enumerate(user_ids):
    output += str(id) + " " + str(index) + "\n"

f = open("user_list.txt", "w")
f.write(output)
f.close()
