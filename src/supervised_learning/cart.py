import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba
        
    def is_leaf(self):
        return self.value is not None or self.proba is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.n_classes_ = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def predict_proba(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            node = self.tree
            while not node.is_leaf():
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.proba)
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value, leaf_proba = self._leaf_value(y)
            return Node(value=leaf_value, proba=leaf_proba)

        best_feature, best_threshold = self._find_best_split(X, y, n_features)

        if best_feature is None or best_threshold is None:
            leaf_value, leaf_proba = self._leaf_value(y)
            return Node(value=leaf_value, proba=leaf_proba)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Ensure children are non-empty
        if not np.any(left_indices) or not np.any(right_indices):
            leaf_value, leaf_proba = self._leaf_value(y)
            return Node(value=leaf_value, proba=leaf_proba)

        left_child = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _find_best_split(self, X, y, n_features):
        best_gain = -1
        best_feature, best_threshold = None, None
        parent_impurity = self._gini_impurity(y)
        n_parent = len(y)

        for feature_idx in range(n_features):
            # Sort data based on the current feature
            sorted_indices = np.argsort(X[:, feature_idx])
            X_sorted, y_sorted = X[sorted_indices], y[sorted_indices]
            
            # Use running counts for efficiency
            left_counts = Counter()
            right_counts = Counter(y)

            for i in range(1, n_parent):
                # Update counts by moving one sample from right to left
                label = y_sorted[i-1]
                left_counts[label] += 1
                right_counts[label] -= 1

                # Don't split if the feature value is the same as the next one
                if X_sorted[i, feature_idx] == X_sorted[i-1, feature_idx]:
                    continue
                
                # Calculate child impurities efficiently
                n_left, n_right = i, n_parent - i
                
                # --- MODIFIED SECTION ---
                # Use a more robust Gini calculation that considers all classes
                
                # Gini for left child
                p_left = np.array([left_counts.get(c, 0) for c in range(self.n_classes_)]) / n_left
                impurity_left = 1.0 - np.sum(p_left**2)

                # Gini for right child
                if n_right > 0:
                    p_right = np.array([right_counts.get(c, 0) for c in range(self.n_classes_)]) / n_right
                    impurity_right = 1.0 - np.sum(p_right**2)
                else:
                    impurity_right = 0.0 # No impurity if the node is empty
                # --- END MODIFIED SECTION ---

                # Weighted average of children's impurity
                child_impurity = (n_left / n_parent) * impurity_left + (n_right / n_parent) * impurity_right
                
                gain = parent_impurity - child_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    # The threshold is the midpoint between two consecutive unique values
                    best_threshold = (X_sorted[i, feature_idx] + X_sorted[i-1, feature_idx]) / 2

        return best_feature, best_threshold
    
    def _leaf_value(self, y):
        counts = Counter(y)
        if not counts: return None, np.zeros(self.n_classes_)
        most_common = counts.most_common(1)[0][0]
        proba = np.array([counts.get(i, 0) / len(y) for i in range(self.n_classes_)])
        return most_common, proba

    def _gini_impurity(self, y):
        m = len(y)
        if m == 0:
            return 0
        p = np.bincount(y, minlength=self.n_classes_) / m
        return 1 - np.sum(p ** 2)

    def _information_gain(self, X, y, feature_idx, threshold):
        # Calculate parent impurity
        parent_impurity = self._gini_impurity(y)

        # Split data and get children
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices
        
        y_left, y_right = y[left_indices], y[right_indices]

        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        # Calculate weighted average of children's impurity
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        child_impurity = (n_left / n) * self._gini_impurity(y_left) + \
                         (n_right / n) * self._gini_impurity(y_right)

        # Information gain is the reduction in impurity
        return parent_impurity - child_impurity