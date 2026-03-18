import numpy as np

class  CustomeMixedNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._classes = None
        self._priors = None

        self.binrary_cols = []
        self.continous_cols = []

        self._mean = None
        self._var = None

        self._feature_probs =None

    def fit(self, X, y):

        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self.binrary_cols = []
        self.continous_cols = []
        #phân loại
        for i in range(n_features):
            unique_vals = np.unique(X[:,i])
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0,1}):
                self.binrary_cols.append(i)
            else:
                self.continous_cols.append(i)

        self._priors = np.zeros(n_classes, dtype=float)
        if self.continous_cols:
            self._mean = np.zeros((n_classes, len(self.continous_cols)), dtype=float)
            self._var =  np.zeros((n_classes, len(self.continous_cols)), dtype=float)
        if self.binrary_cols:
            self._feature_probs = np.zeros((n_classes, len(self.binrary_cols)), dtype=float) 
        
        #Huấn luyên
        for idx, c in enumerate(self._classes):
            X_c = X[y==c]
            self._priors[idx] = X_c.shape[0]/float(n_samples)

            if self.continous_cols:
                X_c_count = X_c[:, self.continous_cols]
                self._mean[idx, :] = X_c_count.mean(axis=0)
                self._var[idx, :]  = X_c_count.var(axis=0)
            
            if self.binrary_cols:
                X_c_bin = X_c[:, self.binrary_cols]
                feature_counts = X_c_bin.sum(axis=0)
                self._feature_probs[idx, :] = (feature_counts+self.alpha)/(X_c.shape[0] + 2*self.alpha)
        

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            log_posterior = np.log(self._priors[idx])
            
            if self.continous_cols:
                x_cont = x[self.continous_cols]
                mean = self._mean[idx]
                var = self._var[idx]
                eps = 1e-4

                numerator = np.exp(-((x_cont - mean) ** 2) / (2 * (var + eps)))
                denominator = np.sqrt(2 * np.pi * (var + eps))
                pdf_vals = numerator / denominator

                log_posterior += np.sum(np.log(pdf_vals +eps))

            if self.binrary_cols:
                x_bin = x[self.binrary_cols]
                p = self._feature_probs[idx]

                log_prob_bin = x_bin*np.log(p) + (1-x_bin)*np.log(1-p)
                log_posterior += np.sum(log_prob_bin)
            
            posteriors.append(log_posterior)

        return self._classes[np.argmax(posteriors)]