import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        #error if k <= 0
        assert k > 0, "k must be greater than 0"

        #error if max_iter <= 0
        assert max_iter > 0, "max iter must be greater than 0"

        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.error = None


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        
        self._initialize_centroids(mat)
        iter_count = 0
        
        while iter_count < self.max_iter:
            
            #assign points to clusters
            distances = cdist(mat, self.centroids, metric='euclidean')
            self.clusters = np.argmin(distances,axis=1)
            
            #calculate centroid of new clusters
            new_centroids = np.zeros((self.k, mat.shape[1]))
            for cluster in range(self.k):

                cluster_points = mat[np.where(self.clusters==cluster)]                 
                if len(cluster_points) == 0:
                    new_centroids[cluster] = self.centroids[cluster]
                else:
                    new_centroids[cluster] = np.mean(cluster_points, axis=0)
                
            self.centroids = new_centroids
            
            min_distances = distances[np.arange(len(distances)), self.clusters]
            new_error = np.mean(min_distances**2)
            
            if iter_count>0 and abs(self.error-new_error) < self.tol:
                self.error = new_error
                break

            self.error = new_error
            iter_count += 1
                    

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        assert mat.size, "Cannot pass an empty matrix"
        assert mat.shape[0] >=self.k, "Must have at least k rows in matrix"
        
        self.fit(mat)
        return self.clusters


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if not hasattr(self, "error"):
            raise AttributeError("No error attribute - have you run predict?")

        return self.error
            
        
    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids
            
        
    def _initialize_centroids(self, mat,):
        
        """Randomly select k points in matrix to be initial locations of centroids"""
        
        self.centroids = mat[np.random.choice(np.arange(mat.shape[0]), size=self.k, replace=False)]
