import torch

from .base_model import BaseModel


class GreedyGapModel(BaseModel):
    """
    Gap heuristic (default) or Rokicki heuristic (mode='rokicki').

    Shared convention for all modes:
    - gaps/breaks are computed on p + [N] (plate-only, no top sentinel).

    mode="gap":
    - score = gap_1 + w2 * gap_2 + w3 * gap_3,
      where gap_1 is plate-only adjacent gaps on p + [N],
      and gap_2/gap_3 count violations of |p[i+k]-p[i]|=k on p itself.
    - Optional: locked_gap_add=1 adds 1 to gap_1 in locked non-goal states.
    - Optional: use_dual_max takes max(score(p), score(dual(p))).

    mode="rokicki":
    - primary key is breaks = gap_1 on p + [N].
    - ties are broken by preferring more singletons (break on both sides).
      Encoded for minimization as score = breaks * (N + 1) - singletons.
    - Optional: locked_gap_add=1 adjusts breaks as in the shared rule above.
    - Optional: use_dual_max takes max(score(p), score(dual(p))).

    dual(p) is the inverse permutation.
    Link: https://tomas.rokicki.com/pancake/
    """

    def __init__(
        self,
        state_size: int,
        w2: float = 0.25,
        w3: float = 0.125,
        mode: str = "gap",
        locked_gap_add: int = 0,
        use_dual_max: bool = False,
    ):
        self.state_size = int(state_size)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.mode = str(mode).lower()
        self.locked_gap_add = int(locked_gap_add)
        self.use_dual_max = bool(use_dual_max)

        if self.mode not in ("gap", "rokicki"):
            raise ValueError(f"mode must be 'gap' or 'rokicki', got {mode!r}")
        if self.locked_gap_add not in (0, 1):
            raise ValueError(
                "locked_gap_add must be 0 or 1, got "
                f"{self.locked_gap_add}"
            )

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        return None

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got ndim={X.ndim}")

        values = X.to(dtype=torch.int16)
        score = self._predict_one(values)
        if not self.use_dual_max:
            return score

        dual_values = self._inverse_permutation(values)
        dual_score = self._predict_one(dual_values)
        return torch.maximum(score, dual_score)

    def _predict_one(self, values: torch.Tensor) -> torch.Tensor:
        if self.mode == "rokicki":
            return self._predict_rokicki(values)
        return self._predict_gap(values)

    def _plate_gaps_int32(self, values: torch.Tensor) -> torch.Tensor:
        state_size = self.state_size
        plate = torch.full(
            (values.size(0), 1),
            state_size,
            device=values.device,
            dtype=values.dtype,
        )
        extended = torch.cat([values, plate], dim=1)
        diffs = (extended[:, 1:] - extended[:, :-1]).abs()
        return (diffs != 1).sum(dim=1, dtype=torch.int32)

    def _locked_mask(self, values: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
        state_size = self.state_size
        if state_size < 2:
            return torch.zeros((values.size(0),), device=values.device,
                               dtype=torch.bool)

        top = values[:, :1]
        plate = torch.full(
            (values.size(0), 1),
            state_size,
            device=values.device,
            dtype=values.dtype,
        )
        old_last = values[:, 1:]
        suffix = torch.cat([values[:, 2:], plate], dim=1)

        old_gap = (old_last - suffix).abs() != 1
        new_gap = (top - suffix).abs() != 1
        has_decrease = (old_gap & ~new_gap).any(dim=1)

        return (gaps > 0) & ~has_decrease

    def _predict_gap(self, values: torch.Tensor) -> torch.Tensor:
        state_size = self.state_size

        gap_1_i32 = self._plate_gaps_int32(values)
        if self.locked_gap_add == 1:
            locked = self._locked_mask(values, gap_1_i32)
            gap_1_i32 = gap_1_i32 + locked.to(torch.int32)

        gap_1 = gap_1_i32.to(torch.float32)
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

    def _predict_rokicki(self, values: torch.Tensor) -> torch.Tensor:
        state_size = self.state_size

        breaks_i32 = self._plate_gaps_int32(values)
        if self.locked_gap_add == 1:
            locked = self._locked_mask(values, breaks_i32)
            breaks_i32 = breaks_i32 + locked.to(torch.int32)

        plate = torch.full(
            (values.size(0), 1),
            state_size,
            device=values.device,
            dtype=values.dtype,
        )
        extended = torch.cat([values, plate], dim=1)
        diffs = (extended[:, 1:] - extended[:, :-1]).abs()
        breaks_mask = diffs != 1

        singletons_mask = breaks_mask[:, :-1] & breaks_mask[:, 1:]
        singletons_i32 = singletons_mask.sum(dim=1, dtype=torch.int32)

        score_i32 = breaks_i32 * (state_size + 1) - singletons_i32
        return score_i32.to(torch.float32)

    def _inverse_permutation(self, values: torch.Tensor) -> torch.Tensor:
        state_size = self.state_size
        positions = torch.arange(
            state_size,
            device=values.device,
            dtype=torch.int16,
        ).unsqueeze(0).expand(values.size(0), -1)

        inverse = torch.empty_like(values, dtype=torch.int16)
        inverse.scatter_(1, values.to(dtype=torch.long), positions)
        return inverse

    def debug_score_components(self, X: torch.Tensor) -> dict:
        """Return score components for the first row of X (for debug)."""
        if X.ndim == 1:
            X = X.unsqueeze(0)
        values = X[:1].to(dtype=torch.int16)
        state_size = self.state_size
        out = {"mode": self.mode}

        if self.mode == "gap":
            gap_1_i32 = self._plate_gaps_int32(values)
            locked = self._locked_mask(values, gap_1_i32) if self.locked_gap_add == 1 else None
            gap_1 = int(gap_1_i32[0].item())
            locked_add = int(locked[0].item()) if locked is not None else 0
            out["gap_1"] = gap_1
            out["locked_gap_add"] = locked_add
            score_d = gap_1 + locked_add
            if state_size >= 3 and self.w2 > 0.0:
                diffs_2 = (values[:, 2:] - values[:, :-2]).abs()
                gap_2 = int((diffs_2 != 2).sum(dim=1)[0].item())
                out["gap_2"] = gap_2
                score_d = score_d + self.w2 * gap_2
            else:
                out["gap_2"] = 0
            if state_size >= 4 and self.w3 > 0.0:
                diffs_3 = (values[:, 3:] - values[:, :-3]).abs()
                gap_3 = int((diffs_3 != 3).sum(dim=1)[0].item())
                out["gap_3"] = gap_3
                score_d = score_d + self.w3 * gap_3
            else:
                out["gap_3"] = 0
            out["base_direct"] = float(score_d)
            if self.use_dual_max:
                dual_v = self._inverse_permutation(values)
                out["base_dual"] = float(self._predict_one(dual_v)[0].item())
                out["base"] = max(out["base_direct"], out["base_dual"])
            else:
                out["base_dual"] = None
                out["base"] = out["base_direct"]
        else:
            breaks_i32 = self._plate_gaps_int32(values)
            if self.locked_gap_add == 1:
                locked = self._locked_mask(values, breaks_i32)
                breaks_i32 = breaks_i32 + locked.to(torch.int32)
            plate = torch.full(
                (1, 1), state_size, device=values.device, dtype=values.dtype,
            )
            extended = torch.cat([values, plate], dim=1)
            diffs = (extended[:, 1:] - extended[:, :-1]).abs()
            breaks_mask = diffs != 1
            singletons_i32 = (breaks_mask[:, :-1] & breaks_mask[:, 1:]).sum(dim=1, dtype=torch.int32)
            out["breaks"] = int(breaks_i32[0].item())
            out["singletons"] = int(singletons_i32[0].item())
            out["base"] = float(breaks_i32[0].item() * (state_size + 1) - singletons_i32[0].item())
            if self.use_dual_max:
                dual_v = self._inverse_permutation(values)
                out["base_dual"] = float(self._predict_one(dual_v)[0].item())
                out["base"] = max(out["base"], out["base_dual"])
            else:
                out["base_dual"] = None
        return out

    def save(self, path: str) -> None:
        raise NotImplementedError("GreedyGapModel has no parameters to save.")

    def load(self, path: str) -> None:
        raise NotImplementedError("GreedyGapModel has no parameters to load.")