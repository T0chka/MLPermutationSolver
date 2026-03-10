"""
Solver factory with registry-based solver construction.
Caller must provide a ready adapter.
"""

from typing import Any, Callable, Dict

from src.puzzles import PuzzleSpec
from src.solvers.beam_solver import (
    BeamSolver,
    IndependentFrontierScheduler,
    SolutionObjective,
)
from src.solvers.pancake_exact_solver import PancakeExactSolver

SolverBuilder = Callable[..., Any]

_SOLVER_REGISTRY: Dict[str, SolverBuilder] = {
    "beam": BeamSolver,
    "pancake_exact": PancakeExactSolver,
}

_EXACT_OPTIONS = frozenset({
    "exact_verify_margin",
    "exact_incumbent_solver_type",
})

_CORE_OPTIONS = frozenset({"objective", "backward_mode"})


def make_solver(
    puzzle_spec: PuzzleSpec,
    solver_type: str,
    adapter: Any,
    **kwargs,
):
    """Create a solver from registry. Caller must pass adapter from get_adapter()."""
    if solver_type not in _SOLVER_REGISTRY:
        supported = ", ".join(sorted(_SOLVER_REGISTRY))
        raise ValueError(f"Unknown solver_type={solver_type!r}; supported: {supported}")
    if adapter is None:
        raise ValueError("adapter is required; create it via get_adapter()")

    if solver_type == "pancake_exact":
        if puzzle_spec.puzzle_type != "pancake":
            raise ValueError(
                "solver_type='pancake_exact' only supported for "
                f"puzzle_type='pancake'; got {puzzle_spec.puzzle_type!r}"
            )

        incumbent_type = kwargs.get("exact_incumbent_solver_type", "beam")
        if incumbent_type == "pancake_exact":
            raise ValueError("exact incumbent cannot be pancake_exact")

        incumbent_kw = {
            k: v for k, v in kwargs.items()
            if k not in _EXACT_OPTIONS
        }
        incumbent_solver = make_solver(
            puzzle_spec, incumbent_type, adapter, **incumbent_kw
        )
        exact_kw = {
            "puzzle_spec": puzzle_spec,
            "device": puzzle_spec.solved_state.device,
            "adapter": adapter,
            "incumbent_solver": incumbent_solver,
            "exact_verify_margin": kwargs.get("exact_verify_margin", 1),
            "verbose": kwargs.get("verbose", 0),
        }
        return PancakeExactSolver(**exact_kw)

    solver_kw = {
        k: v for k, v in kwargs.items()
        if k not in _EXACT_OPTIONS and k not in _CORE_OPTIONS
    }
    solver_kw["puzzle_spec"] = puzzle_spec
    solver_kw["adapter"] = adapter
    if solver_type == "beam":
        backward_mode = kwargs.get("backward_mode", "off")
        objective_name = kwargs.get("objective", "shortest")
        solver_kw["backward_mode"] = backward_mode
        solver_kw["scheduler"] = IndependentFrontierScheduler(
            backward_mode=backward_mode
        )
        solver_kw["objective"] = SolutionObjective(objective=objective_name)
    return _SOLVER_REGISTRY[solver_type](**solver_kw)
