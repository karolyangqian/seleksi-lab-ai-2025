import numpy as np
from scipy import stats

class KNNClassifier:
    def __init__(self, k:int=5, distance_method:str="euclidean", p:int=3):
        """
        Parameters
        ----------
        k : int, optional
            Jumlah neighbor
        distance_method : str, optional
            Metode distance yang digunakan (default: "euclidean").\n
            Opsi:
            - "euclidean"
            - "manhattan" 
            - "minkowski"
        """
        self.k = k
        self.distance_method = distance_method
        self.p = p
        self.proba = None
        
    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            Fitur input fitting model.
        y : np.ndarray
            Label target fitting model.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray
            Fitur input untuk prediksi.

        Returns
        -------
        np.ndarray
            Label hasil prediksi.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Calculate distances as ndarray with shape (n_X, n_X_train)
        if self.distance_method == "euclidean":
            distances = self.euclidean_distance(X, self.X_train)
        elif self.distance_method == "manhattan":
            distances = self.manhattan_distance(X, self.X_train)
        elif self.distance_method == "minkowski":
            distances = self.minkowski_distance(X, self.X_train, self.p)
        
        # Take k nearest neighbors for each sample (used argpartition because it has O(n) complexity compared to sort that has O(n log n))
        k_nearest_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]

        if self.y_train.ndim == 1:
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            classes = np.unique(self.y_train)
            
            # self.proba will have shape (n_samples, n_classes)
            self.proba = np.array([
                np.mean(k_nearest_labels == c, axis=1) for c in classes
            ]).T
            
            predictions = classes[np.argmax(self.proba, axis=1)]

        elif self.y_train.ndim > 1:
            n_samples = X.shape[0]
            n_labels = self.y_train.shape[1]
            predictions = np.zeros((n_samples, n_labels))
            self.proba = None

            for i in range(n_labels):
                k_nearest_labels = self.y_train[k_nearest_indices, i]
                classes = np.unique(self.y_train[:, i])
                
                proba_current_label = np.array([
                    np.mean(k_nearest_labels == c, axis=1) for c in classes
                ]).T
                
                predictions[:, i] = classes[np.argmax(proba_current_label, axis=1)]

                if i == 0:
                    self.proba = proba_current_label
        else:
            predictions = np.zeros(X.shape[0])
            self.proba = np.zeros((X.shape[0], 1))

        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Returns
        -------
        np.ndarray
            The class probabilities of the input samples. Shape (n_samples, n_classes)
        """
        self.predict(X)
        return self.proba

    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray with shape (a, b)
        """
        return np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2, axis=2))

    def manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray with shape (a, b)
        """
        return np.sum(np.abs(a[:, np.newaxis, :] - b[np.newaxis, :, :]), axis=2)

    def minkowski_distance(self, a: np.ndarray, b: np.ndarray, p: int = 3) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray with shape (a, b)
        """
        return np.sum(np.abs(a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** p, axis=2) ** (1 / p)
