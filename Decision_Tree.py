import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right = None, gain = None, value = None):
        #khởi tạo nút của cât quyết định
        self.feature_index = feature_index  # cột nào đang xét
        self.threshold = threshold          # Ngưỡng chia
        self.left = left                    # Nhánh con bên trái
        self.right = right                  # Nhánh con bên phải
        self.gain = gain                    # Độ giảm vấn đục

        self.value = value                  # nếu có value, thì là nút lá

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, criterion='gini', max_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.root = None
    
    def fit(self,X,y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self.root = self._build_tree(X,y,depth = 0)

    def _build_tree(self, X,y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels ==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_split = self._best_split(X, y, n_features)
        if best_split and best_split['gain']>0:
            left_tree = self._build_tree(X[best_split['left_indices'], :], y[best_split['left_indices']], depth + 1)
            right_tree = self._build_tree(X[best_split['right_indices'], :], y[best_split['right_indices']], depth + 1)

            return Node(
                feature_index=best_split['feature_index'],
                threshold=best_split['threshold'],
                left=left_tree,
                right=right_tree,
                gain=best_split['gain']
            )
        
        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)




    def _best_split(self, X,y,n_features):
        best_split_dict = {}
        max_gain = -1

        if self.max_features == 'sqrt':
            num_features = int(np.sqrt(n_features))
        else:
            num_features = n_features
        
        feature_indices = np.random.choice(n_features, num_features, replace=False)

        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_indices, right_indices = self._split(X_column, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                if self.criterion == 'gini':
                    gain = self._information_gain(y,y_left, y_right)
                elif self.criterion == 'entropy':
                    gain = self._information_gain_ID3(y, y_left, y_right) # ID3
                elif self.criterion == 'ratio':
                    gain = self._gain_ratio(y, y_left, y_right)
                
                if gain>max_gain:
                    max_gain = gain
                    best_split_dict = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'gain': gain
                    }
        return best_split_dict
    #===================================hàm hỗ trợ=======================
    #hỗ trợ cho CART=======================================
    def _calculate_gini(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = len(y[y==cls])/len(y)
            gini -= p**2
        return gini
    
    def _information_gain(self, y, y_left, y_right):
        p_left = len(y_left)/len(y)
        p__right = len(y_right)/len(y)
        gini_parent = self._calculate_gini(y)
        gini_children = p_left*self._calculate_gini(y_left) + p__right*self._calculate_gini(y_right)
        return gini_parent-gini_children
    
    #ho tro cho ID3================================================================
    def _calculate_entropy(self, y):
        classes = np.unique(y)
        entropy = 0.0
        for cls in classes:
            p = len(y[y==cls])/len(y)
            if p>0:
                entropy -= p* np.log2(p)
        return entropy
    
    def _information_gain_ID3(self, y, y_left, y_right):
        p_left = len(y_left)/len(y)
        p__right = len(y_right)/len(y)
        entropy_parent = self._calculate_entropy(y)
        entropy_children = p_left*self._calculate_entropy(y_left) + p__right*self._calculate_entropy(y_right)
        return entropy_parent - entropy_children

    #ho tro cho C4.5 (GainRatio)=================================
    def _gain_ratio(self,y,y_left, y_right):
        ig = self._information_gain_ID3(y, y_left, y_right) 

        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)

        split_info = 0.0
        if p_left > 0: 
            split_info -= p_left * np.log2(p_left)
        if p_right > 0: 
            split_info -= p_right * np.log2(p_right)

        if split_info == 0:
            return 0
        return ig/split_info

    #ham ho tro chung=============================================
    def _split(self, X_column, threshold):
        left_indices = np.argwhere(X_column<= threshold).flatten()
        right_indices = np.argwhere(X_column>threshold).flatten()
        return left_indices, right_indices
    
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node:Node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    

    
    