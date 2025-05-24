from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    """Abstract base class for permutation solving models"""
    
    @abstractmethod
    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the model on the given data"""
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict distances for given states"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass 