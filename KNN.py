import numpy as np

class CustomKNN:
    def __init__(self, k=5, p=2, weights='uniform'):
        self.k = k
        self.p = p
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X.astype(float))
        self.y_train = np.array(y.astype(float))

        if self.p == 'mahalanobis':
            cov_matrix = np.cov(self.X_train, rowvar = False)
            self.inv_cov_matrix = np.linalg.pinv(cov_matrix)


    def _predic_single(self,x):
        if self.p == 2:
            distances = np.sqrt(np.sum((self.X_train - x.astype(float))**2, axis = 1))

        elif self.p == 1:
            distances = np.sum(np.abs(self.X_train - x.astype(float)), axis=1)
        
        elif self.p == 'mahalanobis':
            diff = self.X_train - x.astype(float)
            left_term = np.dot(diff, self.inv_cov_matrix)
            distances = np.sqrt(np.sum(left_term * diff, axis=1))
        else:
            raise ValueError('Chưa update các dạng khác')

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        if self.weights == 'uniform':
           unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
           return unique_labels[np.argmax(counts)]
        elif self.weights == 'distance':
            k_nearest_distances = distances[k_indices]
            vote_weights = 1/(k_nearest_distances+1e-5)
            class_0_weight = np.sum(vote_weights[k_nearest_labels == 0])
            class_1_weight = np.sum(vote_weights[k_nearest_labels == 1])
            
            return 1 if class_1_weight > class_0_weight else 0
    
    def predict(self, X):
        X_array = np.array(X.astype(float))
        predictions = [self._predic_single(x) for x in X_array]
        return np.array(predictions)