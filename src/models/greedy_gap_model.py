import torch

from .base_model import BaseModel


class GreedyGapModel(BaseModel):
    """Return a gap-based distance proxy for pancake permutations."""

    def __init__(self, state_size: int, w2: float = 0.25, w3: float = 0.125):
        self.state_size = int(state_size)
        self.w2 = float(w2)
        self.w3 = float(w3)

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        return None

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got ndim={X.ndim}")

        state_size = self.state_size
        values = X.to(dtype=torch.int16)

        left = torch.full((values.size(0), 1), -1, device=X.device, dtype=torch.int16)
        right = torch.full(
            (values.size(0), 1),
            state_size,
            device=X.device,
            dtype=torch.int16,
        )
        augmented = torch.cat([left, values, right], dim=1)

        diffs = (augmented[:, 1:] - augmented[:, :-1]).abs()
        gap_1 = (diffs != 1).sum(dim=1, dtype=torch.int32).to(torch.float32)

        score = gap_1
        if state_size >= 3 and self.w2 > 0.0:
            diffs_2 = (values[:, 2:] - values[:, :-2]).abs()
            gap_2 = (diffs_2 != 2).sum(dim=1, dtype=torch.int32).to(torch.float32)
            score = score + self.w2 * gap_2
        if state_size >= 4 and self.w3 > 0.0:
            diffs_3 = (values[:, 3:] - values[:, :-3]).abs()
            gap_3 = (diffs_3 != 3).sum(dim=1, dtype=torch.int32).to(torch.float32)
            score = score + self.w3 * gap_3

        return score

    def save(self, path: str) -> None:
        raise NotImplementedError("GreedyGapModel has no parameters to save.")

    def load(self, path: str) -> None:
        raise NotImplementedError("GreedyGapModel has no parameters to load.")