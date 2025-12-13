from qubit_ml.utils.logger import get_logger
from sklearn.cluster import KMeans
import numpy as np

log = get_logger(__name__)

class KMeanClustering():
    def __init__(self, k:int=2, ) -> None:
        self.kmeans = KMeans(n_clusters=k)
        
        log.debug("k=%d", k)
        log.debug("K means clustering initialized")
    
    def fit(self, X:np.ndarray):
        print(f"X[{X.shape}]")
        
        log.debug("Fitting the model")
        
        self.kmeans.fit(X)
        
        log.debug("Fitting completed")
    
    def predict(self, X:np.ndarray)->np.ndarray:
        return self.kmeans.predict(X)