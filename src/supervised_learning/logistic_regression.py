import numpy as np

class LogisticRegressionClassifier:
    def __init__(self,
                 learning_rate=0.1,
                 max_iter=100,
                 regularization_term='l1',
                 lambda_reg=0.1,
                 class_weight=None):
      self.learning_rate = learning_rate
      self.max_iter = max_iter
      self.regularization_term = regularization_term
      self.lambda_reg = lambda_reg
      self.class_weight = class_weight
      self.W = None
      self.b = None
      
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, w, b):
        m = len(y)
        z = X @ w + b
        g = self.sigmoid(z)

        # To avoid log(0) errors
        epsilon = 1e-15
        g = np.clip(g, epsilon, 1 - epsilon)

        cost = (-1 / m) * np.sum(y * np.log(g) + (1 - y) * np.log(1 - g))

        # Add regularization term
        if self.regularization_term == 'l2':
            cost += (self.lambda_reg / (2 * m)) * np.sum(w**2)
        elif self.regularization_term == 'l1':
            cost += (self.lambda_reg / m) * np.sum(np.abs(w))

        return cost

    def gradient_function(self, X, y, w, b, sample_weights):
        m = len(y)
        z = X @ w + b
        g = self.sigmoid(z)
        
        error = g - y
        
        # Apply sample weights to the error
        weighted_error = error * sample_weights
        
        grad_w = (1 / m) * (X.T @ weighted_error)
        grad_b = (1 / m) * np.sum(weighted_error)

        # Add regularization gradient
        if self.regularization_term == 'l2':
            grad_w += (self.lambda_reg / m) * w
        elif self.regularization_term == 'l1':
            grad_w += (self.lambda_reg / m) * np.sign(w)

        return grad_w, grad_b
    
    def gradient_descent(self, X, y, learning_rate, iterations):
        w = np.zeros(X.shape[1])
        b = 0
        
        # Calculate class weights
        sample_weights = np.ones(len(y))
        if self.class_weight is not None:
            if self.class_weight == 'balanced':
                unique_classes, class_counts = np.unique(y, return_counts=True)
                n_samples = len(y)
                n_classes = len(unique_classes)
                class_weights = {}
                for cls, count in zip(unique_classes, class_counts):
                    class_weights[cls] = n_samples / (n_classes * count)
            else:
                class_weights = self.class_weight
            
            # Apply weights to samples
            for i in range(len(y)):
                sample_weights[i] = class_weights[y[i]]

        for i in range(iterations):
            grad_w, grad_b = self.gradient_function(X, y, w, b, sample_weights)
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
            
            # Verbose
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {self.cost_function(X, y, w, b)}")

        self.W = w
        self.b = b
        
    def predict(self, X):
        preds = np.zeros(len(X))
        
        for i in range(len(X)):
            z = np.dot(self.W, X[i]) + self.b
            g = self.sigmoid(z)

            preds[i] = 1 if g > 0.5 else 0

        return preds
    
    def fit(self, X, y):
        self.gradient_descent(X, y, self.learning_rate, self.max_iter)
        
    def predict_proba(self, X):
        z = X @ self.W + self.b
        g = self.sigmoid(z)
        
        prob_class_1 = g.reshape(-1, 1)
        prob_class_0 = 1 - prob_class_1
        
        return np.hstack([prob_class_0, prob_class_1])