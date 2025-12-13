from .base import SVM
from qubit_ml.utils.logger import get_logger
import numpy as np
from sklearn import svm

log = get_logger(__name__)

class NonLinearSVM(SVM):
    # To add parameters here
    def __init__(self) -> None:
        super().__init__()
        self.clf = svm.SVC()
        
        log.debug("Non Linear SVM initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"X[{X.shape}], y[{y.shape}]")
        
        log.debug("Fitting the model")
        
        self.clf.fit(X,y)
        
        log.debug("Fitting completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)