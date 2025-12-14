from abc import ABC, abstractmethod
import numpy as np

class DiscrAnalysis(ABC):
    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray):
        ...
    
    @abstractmethod
    def predict(self, X:np.ndarray)->np.ndarray:
        ...