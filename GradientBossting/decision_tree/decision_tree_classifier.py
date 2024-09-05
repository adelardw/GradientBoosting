from decision_tree.criterions import gini, entropy
from decision_tree.impurity import impurity
import numpy as np


class Node:
    def __init__(self, y=None,
                       l=None,
                       r=None, 
                       split_val=None,
                       split_feature=None):

        self.y = y
        self.l = l
        self.r = r
        self.split_val = split_val
        self.split_feature = split_feature

        if self.y is not None:
            unique, counts = np.unique(self.y, return_counts = True)
            self.prob = dict(zip(unique, counts / self.y.size))
        


class DecisionTreeClassifier:
    def __init__(self, max_depth, min_leaf_samples, criterion):
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.criterion = gini if criterion == 'gini' else entropy

    def search_max_impurity_feature(self, X, y):
        num_features = X.shape[1]

        max_impurity = -np.inf
        for i in range(num_features):
            feature = X[:, i]
            split_values = np.unique(feature)
            max_impurity_feature = -np.inf
            for split_val in split_values:
                left_y = y[feature < split_val]
                right_y = y[feature >= split_val]
                impurity_result = impurity(left_y, right_y, self.criterion)

                if impurity_result > max_impurity_feature:
                    max_impurity_feature = impurity_result
                    best_split = split_val
                
            if max_impurity_feature > max_impurity:
                max_impurity = max_impurity_feature
                best_split_value = best_split
                best_split_feature = i
            
        return best_split_value, best_split_feature
    
    def build_tree(self, X, y ,depth=0):
        depth += 1

        if X.shape[0] >= self.min_leaf_samples:

            best_split_value, best_split_feature = self.search_max_impurity_feature(X, y)
            
            mask = X[:, best_split_feature] < best_split_value

            left_X = X[mask]
            right_X = X[~mask]
            left_y = y[mask]
            right_y = y[~mask]

            if self.max_depth is not None:
                if len(left_X) == 0 or len(right_X) == 0 or depth >=self.max_depth:
                    return Node(y=y,split_val=best_split_value,split_feature=best_split_feature)
            
            else:
                if len(left_X) == 0 or len(right_X) == 0:
                    return Node(y=y,split_val=best_split_value,split_feature=best_split_feature)
            


            return Node(l=self.build_tree(left_X, left_y, depth),
                        r=self.build_tree(right_X, right_y, depth),
                        split_val=best_split_value,
                        split_feature=best_split_feature)
        
        return Node(y=y)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.labels = np.unique(self.y)
        self.tree = self.build_tree(self.X, self.y, 0)

    def tree_traveling(self, X, indecies, tree, proba):
        
        if tree.y is not None:
            for i in indecies:
                proba[i] = tree.prob

            return proba
        else:
            
            mask = X[:, tree.split_feature] < tree.split_val

            left_X = X[mask]
            right_X = X[~mask]
            left_indecies = indecies[mask]
            right_indecies = indecies[~mask]

            self.tree_traveling(left_X,left_indecies, tree.l, proba) 
            self.tree_traveling(right_X,right_indecies, tree.r, proba) 
        
        return proba


    def predict_proba(self, X):

        proba = [dict.fromkeys(self.labels)] * X.shape[0]
        return self.tree_traveling(X, np.arange(X.shape[0]), self.tree, proba)
    

    def predict(self, X):

        out = self.predict_proba(X)
        return np.array([max(p, key = lambda k:p[k]) for p in out])
    
