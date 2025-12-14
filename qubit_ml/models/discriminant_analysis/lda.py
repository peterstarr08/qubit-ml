from numpy import ndarray
from .base import DiscrAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA(DiscrAnalysis):
    def __init__(self) -> None:
        self.clf = LinearDiscriminantAnalysis()
    
    def fit(self, X: ndarray, y: ndarray):
        self.clf.fit(X,y)
    
    def predict(self, X: ndarray) -> ndarray:
        return self.clf.predict(X)