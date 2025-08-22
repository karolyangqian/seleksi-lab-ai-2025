from matplotlib.pylab import Literal
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, criterion:Literal['gini', 'entropy']= 'gini', random_state=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        if criterion not in ('gini', 'entropy'):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.random_state = np.random.RandomState(random_state) if random_state is not None else np.random
        self.root = None
        self.n_features_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        if self.n_features is None:
            self._feat_subset_size = self.n_features_
        else:
            self._feat_subset_size = min(self.n_features, self.n_features_)
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (n_samples < self.min_samples_split) or (depth >= self.max_depth) or (num_labels == 1):
            leaf_value = self._most_common_label(y)
            proba = self._leaf_proba(y)
            return Node(value=leaf_value, proba=proba)

        feat_idxs = self.random_state.choice(n_feats, self._feat_subset_size, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # if no valid split found -> make leaf
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            proba = self._leaf_proba(y)
            return Node(value=leaf_value, proba=proba)

        # partition
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_impurity = self._impurity(y)

        n = len(y)
        for feat in feat_idxs:
            X_col = X[:, feat]
            sort_idx = np.argsort(X_col)
            X_sorted = X_col[sort_idx]
            y_sorted = y[sort_idx]

            # counts for cumulative left and right
            right_counter = Counter(y_sorted)
            left_counter = Counter()

            # iterate possible split positions (only where value changes)
            for i in range(1, n):
                label = y_sorted[i-1]
                left_counter[label] += 1
                right_counter[label] -= 1
                if right_counter[label] == 0:
                    del right_counter[label]

                # skip if same feature value -> no meaningful split
                if X_sorted[i] == X_sorted[i-1]:
                    continue

                n_left = i
                n_right = n - i
                if n_left < self.min_samples_split or n_right < self.min_samples_split:
                    continue

                # compute impurity for children
                impurity_left = self._impurity_from_counts(left_counter, n_left)
                impurity_right = self._impurity_from_counts(right_counter, n_right)

                # weighted impurity
                child_impurity = (n_left / n) * impurity_left + (n_right / n) * impurity_right
                information_gain = parent_impurity - child_impurity

                if information_gain > best_gain:
                    best_gain = information_gain
                    split_idx = feat
                    # threshold: midpoint between the two adjacent values
                    split_thresh = (X_sorted[i] + X_sorted[i-1]) / 2.0

        return split_idx, split_thresh

    def _impurity(self, y):
        # compute impurity of array y
        counter = Counter(y)
        n = len(y)
        return self._impurity_from_counts(counter, n)

    def _impurity_from_counts(self, counter, n):
        if n == 0:
            return 0.0
        probs = np.array([counter.get(c, 0) for c in self.classes_], dtype=float) / n
        if self.criterion == 'gini':
            return 1.0 - np.sum(probs ** 2)
        else:  # entropy
            nz = probs[probs > 0]
            return -np.sum(nz * np.log(nz))

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _leaf_proba(self, y):
        counter = Counter(y)
        total = len(y)
        proba = np.array([counter.get(c, 0) / total for c in self.classes_], dtype=float)
        return proba

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X):
        X = np.array(X)
        probs = np.array([self._traverse_proba(x, self.root) for x in X])
        return probs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _traverse_proba(self, x, node):
        if node.is_leaf_node():
            return node.proba
        if x[node.feature] <= node.threshold:
            return self._traverse_proba(x, node.left)
        return self._traverse_proba(x, node.right)