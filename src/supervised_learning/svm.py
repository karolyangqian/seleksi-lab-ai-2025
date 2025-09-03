import numpy as np

class SVMClassifier:
    def __init__(self, learning_rate=0.001, C=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        y_ = np.where(y == 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                correct_prediction = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if correct_prediction:
                    self.w -= self.learning_rate * (2 * self.C * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.C * self.w - x_i * y_[idx])
                    self.b += self.learning_rate * y_[idx]
                    
        return self
    
    def predict(self, X):
        linear_output = self.decision_function(X)
        return np.where(linear_output >= 0, 1, 0)
    
    def decision_function(self, X):
        return np.dot(X, self.w) + self.b