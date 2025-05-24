import torch
from catboost import CatBoostRegressor
from .base_model import BaseModel

class CatBoostModel(BaseModel):
    """CatBoost implementation for permutation solving"""
    
    def __init__(
        self,
        n_estimators: int = 2000,
        learning_rate: float = 0.05
    ):
        self.model = CatBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            task_type="GPU",
            thread_count=-1,
            verbose=0
        )
    
    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the model on the given data"""
        self.model.fit(X.cpu().numpy(), y.cpu().numpy())
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict distances for given states"""
        predictions = self.model.predict(X.cpu().numpy())
        return torch.tensor(predictions, device=X.device)
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        self.model.save_model(path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        self.model.load_model(path) 