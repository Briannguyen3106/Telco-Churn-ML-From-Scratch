import numpy as np

class CustomLinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.tol = tol
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        
        #chuyển đổi nhãn từ {0, 1} sang {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            w_old = self.w.copy()
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b)

                if condition >=1:
                    self.w -= self.learning_rate* (2*self.lambda_param*self.w)
                
                else:
                    self.w -= self.learning_rate*(2*self.lambda_param*self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate*y_[idx]
            weight_change = np.linalg.norm(self.w - w_old)
            if weight_change < self.tol:
                break
        
    def predict(self, X):
        X = np.array(X, dtype=float)

        approx = np.dot(X, self.w) - self.b

        predicted_label = np.sign(approx)

        return np.where(predicted_label==-1, 0, 1)