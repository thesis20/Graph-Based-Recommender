import pandas as pd
import random


class DataModifier():
    def __init__(self, data):
        self.data = data

    def add_context_dependency(self, alpha=0.5, beta=0.5):
        """
        Adds a contextual feature for each rating.
        Ratings will be incremented or decremented based on this feature.
        Parameters:
            alpha (float): determines how many ratings should be
                increased or decreased
        """
        context = [random.randint(0, 1) for x in range(len(self.data.index))]
        self.data['context_feature'] = context
        frac = self.data.sample(frac=alpha)

        for index, row in frac.iterrows():
            if row['context_feature'] == 1:
                if int(self.data.loc[index, 'rating']) != 5:
                    self.data.loc[index, 'rating'] += 1
            else:
                if int(self.data.loc[index, 'rating']) != 1:
                    self.data.loc[index, 'rating'] -= 1

        return self.data


data = pd.read_csv('data/ml-100k/ratings.csv')

dm = DataModifier(data)
modified_data = dm.add_context_dependency(alpha=1)
