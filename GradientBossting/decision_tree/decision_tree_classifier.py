from decision_tree.criterions import gini, entropy
from decision_tree.gain import gain
import numpy as np


class Node:
    def __init__(self, left , right, split_val, feature):
        self.left = left
        self.right = right
        self.split_val = split_val
        self.feature = feature


class Leaf:
    def __init__(self, y):
        labels, counts = np.unique(y, return_counts=True)

        self.prob = dict(zip(labels, counts / y.size))


class Module:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict_proba(self):
        pass

    def predict(self):
        super().__call__()
        pass

class DecisionTreeClassifier(Module):
    def __init__(self, max_depth, min_leaf_samples, criterion):
        super().__init__()

        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.criterion = gini if criterion == 'gini' else entropy
        self.mode = mode


    def __max_gain(self, X, y):

        num_features = X.shape[1]
        max_gain = -np.inf

        for i in range(num_features):
            x = X[:, i]
            split_values = np.unique(x)
            max_gain_features = -np.inf

            for split_val in split_values:

                mask = x < split_val
                left_y = y[mask]
                right_y = y[~mask]
                
                gain_result = gain(left_y, right_y, self.criterion)

                if gain_result > max_gain_features:
                    max_gain_features = gain_result
                    best_split = split_val
            

            if max_gain_features > max_gain:
                max_gain = max_gain_features
                feature = i
                split_value = best_split
        

        return max_gain, feature, split_value


    def __build_tree(self, X , y, depth = 0):
        depth += 1
        if len(y) >= self.min_leaf_samples:
            max_gain, feature, split_value =self.__max_gain(X, y)

            mask = X[:, feature] < split_value

            left_y = y[mask]
            right_y = y[~mask]
            left_X = X[mask]
            right_X = X[~mask]

            if self.max_depth is None:
                if len(left_X) == 0 or len(right_X) == 0:
                    return Leaf(y)
            else:
                if len(left_X) == 0 or len(right_X) == 0 or depth >= self.max_depth:
                    return Leaf(y)
            
            return Node(left=self.__build_tree(left_X, left_y, depth),
                        right=self.__build_tree(right_X, right_y, depth),
                        split_val=split_value,
                        feature=feature)
        
        return Leaf(y)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.labels = np.unique(self.y)
        self.tree = self.__build_tree(self.X, self.y, 0)


    def __tree_climb(self, X, indecies, proba, node_or_leaf):
        
        if isinstance(node_or_leaf, Leaf):
            leaf = node_or_leaf
            for i in indecies:
                proba[i] = leaf.prob

            return proba

        else:
            node = node_or_leaf

            mask = X[:, node.feature] < node.split_val
            left_X = X[mask]
            right_X = X[~mask]
            left_indecies = indecies[mask]
            right_indecies = indecies[~mask]

            self.__tree_climb(left_X, left_indecies, proba, node.left)
            self.__tree_climb(right_X, right_indecies, proba, node.right)
        
        return proba
    
    def predict_proba(self, X):
        proba = [dict.fromkeys(self.labels)] * X.shape[0]

        return self.__tree_climb(X, np.arange(X.shape[0]), proba, self.tree)
    
    def predict(self, X):
        return np.array([max(p, key= lambda k: p[k]) for p in self.predict_proba(X)])
    
