import numpy as np

class KNNClassifier:
    def __init__(self, k:int=1, distance_method:str="euclidean"):
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
        
    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            Fitur input fitting model.
        y : np.ndarray
            Label target fitting model.
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray
            Fitur input untuk prediksi.

        Returns
        -------
        np.ndarray
            Label yang diprediksi.
        """
        distances = []
        for i in range(len(self.X_train)):
            if self.distance_method == "euclidean":
                dist = self.euclidean_distance(X, self.X_train[i])
            elif self.distance_method == "manhattan":
                dist = self.manhattan_distance(X, self.X_train[i])
            elif self.distance_method == "minkowski":
                dist = self.minkowski_distance(X, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        labels = [label for _, label in neighbors]
        return np.array(labels)

    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.sqrt(np.sum((a - b) ** 2))
    
    def manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.sum(np.abs(a - b))
    
    def minkowski_distance(self, a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
        return np.sum(np.abs(a - b) ** p) ** (1 / p)
