import numpy as np

class GaussianNaiveBayesClassifier:
    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors_set = False if priors is None else True
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.classes = None
        
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)
        n_classes, n_features = len(self.classes), X.shape[1]
        
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        if not self.priors_set:
            self.priors = np.zeros(n_classes)

        for idx, cls in enumerate(self.classes):
            X_c = X[y == cls]
            
            self.means[idx] = X_c.mean(axis=0)
            self.variances[idx] = X_c.var(axis=0) + self.var_smoothing
            if not self.priors_set:
                self.priors[idx] = X_c.shape[0] / X.shape[0]

        return self
    
    def log_gaussian(self, X):
        num = -0.5 * (X[:, None, :] - self.means) ** 2 / self.variances
        log_prob = num - 0.5 * np.log(2 * np.pi * self.variances)
        return log_prob.sum(axis=2)
    
    def predict(self, X):
        X = np.array(X)
        epsilon = 1e-9

        log_likelihood = self.log_gaussian(X)
        log_prior = np.log(self.priors + epsilon)
        return self.classes[np.argmax(log_likelihood + log_prior, axis=1)]

    def predict_proba(self, X):
        X = np.array(X)
        epsilon = 1e-9

        log_likelihood = self.log_gaussian(X)
        log_prior = np.log(self.priors + epsilon)
        log_posterior = log_likelihood + log_prior

        max_log_posterior = np.max(log_posterior, axis=1, keepdims=True)
        log_posterior_shifted = log_posterior - max_log_posterior

        exp_log_posterior = np.exp(log_posterior_shifted)

        sum_exp = np.sum(exp_log_posterior, axis=1, keepdims=True)
        probabilities = exp_log_posterior / sum_exp

        return probabilities