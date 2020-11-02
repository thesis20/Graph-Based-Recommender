import torch
import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

class LightGCN():

    def load_data(self, training_size=0.8):
        full_data = pd.read_csv('../../data/ml-100k/ratings.csv', sep=',')

        full_data = full_data.drop_duplicates()
        v = full_data[['userId']]
        full_data = full_data[v.replace(v.apply(pd.Series.value_counts)).gt(10).all(1)]

        #y = full_data[['movieId']]
        #full_data = full_data[y.replace(y.apply(pd.Series.value_counts)).gt(10).all(1)]

        # Split into test and training
        msk = np.random.rand(len(full_data)) < training_size
        self.training_data = full_data[msk] # Training data
        self.test_data = full_data[~msk] # Test data

        users = self.training_data.userId.unique()
        items = self.training_data.movieId.unique()
        self.user_count = len(users)
        self.item_count = len(items)
        self.users_dict = {k: v for v, k in enumerate(users)}
        self.items_dict = {k: v for v, k in enumerate(items)}

    def train(self, embedding_size=64):
        user_item_matrix = torch.sparse.Tensor((self.user_count, self.item_count), dtype=np.bool)

        # generate user-item interaction matrix
        for _, row in self.training_data.iterrows():
            userIndex = self.users_dict[row['userId']]
            itemIndex = self.items_dict[row['movieId']]
            user_item_matrix[userIndex, itemIndex] = True

        user_item_count = self.user_count + self.item_count
        adjacency_matrix = lil_matrix((user_item_count, user_item_count), dtype=np.bool)

        for i in range(self.user_count):
            for j in range(self.item_count):
                # insert values of R into adjacency matrix
                adjacency_matrix[i, j + self.user_count] = user_item_matrix[i, j]
                # insert values of R_transposed into adjacency matrix
                adjacency_matrix[j + self.user_count, i] = user_item_matrix[i, j]

        embedding_matrix = lil_matrix((user_item_count, embedding_size), dtype=np.float32)
        diagonal_matrix = lil_matrix((user_item_count, user_item_count), dtype=np.int32)

        for i in range(user_item_count):
            print(adjacency_matrix[i].count_nonzero())
            diagonal_matrix[i, i] = adjacency_matrix[i].count_nonzero()
        
        embedding_matrix = ((diagonal_matrix**0.5)*adjacency_matrix*(diagonal_matrix**0.5))*embedding_matrix

lgcn = LightGCN()
lgcn.load_data()
lgcn.train()