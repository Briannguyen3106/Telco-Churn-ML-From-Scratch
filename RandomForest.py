import numpy as np
from collections import Counter
from Decision_Tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth = 10, min_samples_splits =2, criterion='gini'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_splits = min_samples_splits
        self.criterion = criterion
        self.trees = []
    
    def fit(self, X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_trees):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_splits,
                criterion=self.criterion,
                max_features='sqrt'
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)