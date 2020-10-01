class Node:
    def __init__(self, data, probability=None, left=None, right=None):
        self.data = data
        self.probability = probability
        self.parent = None
        self.depth = None
        self.left = left
        self.right = right
