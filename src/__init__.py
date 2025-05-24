"""MLPermutationSolver: A machine learning approach to permutation sorting problems."""

from .models import BaseModel, XGBoostModel, MLPModel, CatBoostModel
from .solvers.base_solver import BaseSolver
from .solvers.simple_solver import BeamSearchSolver
from .data.random_walks import (
    create_lrx_moves,
    first_visit_random_walks,
    nbt_random_walks,
    random_walks_beam_nbt
)

__version__ = "1.0.0"
__author__ = "Tochka"

__all__ = [
    "BaseModel", "XGBoostModel", "MLPModel", "CatBoostModel",
    "BaseSolver", "BeamSearchSolver",
    "create_lrx_moves", "first_visit_random_walks", 
    "nbt_random_walks", "random_walks_beam_nbt"
]
