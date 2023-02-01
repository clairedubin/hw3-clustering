import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        assert X.size, "Cannot pass an empty X matrix"
        assert y.size, "Cannot pass an empty y matrix"
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"

        scores = np.zeros(X.shape[0])
        
        for i,row in enumerate(X):
            mean_inter_cluster_dists = []
            row_label = y[i]
            
            for label in np.unique(y):

                if label == row_label:
                    cluster_points = X[np.where(np.delete(y,i)==label)]
                    mean_intra_cluster_dist = np.mean(cdist([row], cluster_points))
                else:
                    cluster_points = X[np.where(y==label)]
                    mean_inter_cluster_dists += [np.mean(cdist([row], cluster_points))]

            min_inter_cluster_dist = min(mean_inter_cluster_dists)
            s = (min_inter_cluster_dist - mean_intra_cluster_dist)/max(mean_intra_cluster_dist,min_inter_cluster_dist)

            scores[i] = s
            
        return scores
