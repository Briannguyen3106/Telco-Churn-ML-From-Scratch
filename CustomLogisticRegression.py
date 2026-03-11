import numpy as np
class CustomLogisticRegression:
    def __init__(self, learning_rate = 0.01, num_iterations = 1000, penalty = 'none', lambda_param = 0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.penalty = penalty
        self.lambda_param = lambda_param
        self.weight = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weight)+self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted-y)
        
            # Ridge
            if self.penalty.lower() == 'l2':
                dw += (self.lambda_param/n_samples)* self.weight
            #Lasso
            if self.penalty.lower() == "l1":
                dw += (self.lambda_param/n_samples)* np.sign(self.weight)

            self.weight -= self.learning_rate * dw
            self.bias   -= self.learning_rate * db
    
    def predict(self, X):
        linearn_model = np.dot(X, self.weight) + self.bias
        y_predicted = self._sigmoid(linearn_model)
        return np.array([1 if i>0.5 else 0 for i in y_predicted])
    