
import numpy as np
from decision_tree.decision_tree_classifier import DecisionTreeClassifier

class GradientBoostingClassifier:
    def __init__(self, max_depth, min_leaf_samples, criterion, num_iterations, lr):
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.criterion = criterion
        self.num_iterations = num_iterations
        self.lr = lr
        self.models = []  

    def loss_function(self, y, predict):
        
        exp_pred = np.exp(predict - np.max(predict, axis=1, keepdims=True))
        softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        return -np.mean(np.log(softmax[np.arange(len(y)), y]))

    def grad_loss(self, y, predict):
        
        exp_pred = np.exp(predict - np.max(predict, axis=1, keepdims=True))
        softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        grad = softmax
        grad[np.arange(len(y)), y] -= 1
        return grad

    def fit_base_model(self, X, residuals):
        tree = DecisionTreeClassifier(max_depth=self.max_depth, min_leaf_samples=self.min_leaf_samples, criterion=self.criterion)
        tree.fit(X, residuals)
        return tree

    def fit(self, X, y):

        unique, counts = np.unique(y, return_counts=True)
        self.n_classes = len(unique)
        n_samples = X.shape[0]
        
        
        base_model = np.full((n_samples, self.n_classes), np.log(counts / y.size))

        for t in range(self.num_iterations):            
            residuals = self.grad_loss(y, base_model)
            
            models_iteration = []
            for k in range(self.n_classes):
                tree = self.fit_base_model(X, -residuals[:, k])
                models_iteration.append(tree)
                
                base_model[:, k] += self.lr * tree.predict(X)
            
            self.models.append(models_iteration)
            
            
            loss = self.loss_function(y, base_model)
            print(f'Iteration {t + 1}, Loss: {loss:.4f}')

    def predict_proba(self, X):
        pred = np.zeros((X.shape[0], self.n_classes))
        
        for models_iteration in self.models:
            for k, tree in enumerate(models_iteration):
                pred[:, k] += self.lr * tree.predict(X)
        
        exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        return softmax

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


