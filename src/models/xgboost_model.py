import torch
import cupy as cp
import xgboost as xgb
import torch.utils.dlpack

from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost model that:
      - Trains on GPU (using a DMatrix built from CuPy arrays).
      - Performs inference fully on GPU via inplace_predict().
      - Returns predictions directly as a CUDA torch.Tensor, with no CPU copy.
    """

    def __init__(
        self,
        n_estimators: int = 2000,
        learning_rate: float = 0.07,
        verbose: bool = False
    ):
        # XGBoost training parameters
        self.num_boost_round = n_estimators
        self.params = {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "eta": learning_rate,
            "verbosity": 1 if verbose else 0,
            "objective": "reg:squarederror",
        }
        self.booster = None

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train on GPU using CuPy + DMatrix."""
        # Convert PyTorch (GPU) to CuPy via DLPack (no CPU copy):
        # Ensure X,y are contiguous so .to_dlpack() is well-defined.
        X_cupy = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X.contiguous()))
        y_cupy = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y.contiguous()))

        # Build DMatrix for training
        dtrain = xgb.DMatrix(X_cupy, label=y_cupy)

        # Train XGBoost booster
        self.booster = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
        )

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        GPU-based inference using inplace_predict().
        Returns predictions as a CUDA torch.Tensor with no CPU copy.
        """
        if self.booster is None:
            raise ValueError("Model not trained. Call .train(...) first.")

        # Convert PyTorch CUDA tensor to CuPy array
        X_cupy = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X.contiguous()))

        # Predict in-place on GPU; returns a CuPy array
        preds_cupy = self.booster.inplace_predict(X_cupy)
        # (Make sure X_cupy is 2D: (n_samples, n_features).)

        # Convert CuPy -> PyTorch (GPU) via DLPack zero-copy
        preds_gpu = torch.utils.dlpack.from_dlpack(preds_cupy.toDlpack())

        return preds_gpu

    def save(self, path: str) -> None:
        """Save the XGBoost Booster model to disk."""
        if self.booster is None:
            raise ValueError("No trained model to save.")
        self.booster.save_model(path)

    def load(self, path: str) -> None:
        """Load a trained XGBoost Booster model from disk."""
        self.booster = xgb.Booster()
        self.booster.load_model(path)