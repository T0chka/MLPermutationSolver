# MLPermutationSolver

MLPermutationSolver is a GPU-accelerated PyTorch implementation of machine learning-guided beam search for solving permutation sorting problems on Cayley graphs with LRX generators (L - left shift, R - right shift, and X - swap of the first 2 elements).

This repo is a spinoff from [CayleyPy](https://github.com/cayleypy) by Alexander Chervov and collaborators (applying artificial intelligence methods to the mathematical problem of Cayley graph pathfinding). It represents a parallel implementation of the Beam seach results of testing which were reported in academic publications e.g. [here](https://arxiv.org/abs/2502.18663).

## Key Features

- **Random walk data generation algorithms** for training ML models
- **Beam search algorithms** guided by machine learning predictions
- **Multiple model architectures** (XGBoost, CatBoost, MLP)
- **GPU acceleration** (batching, inference, hashing)
- **Benchmarking, profiling, and evaluation tools**

## Installation

### Setup
```bash
git clone https://github.com/T0chka/MLPermutationSolver.git
cd MLPermutationSolver
uv venv
uv sync
```

## Quick Start

To run the full pipeline end-to-end (generate random-walk training data, trains an ML model,
and then solve target permutations with `BeamSearchSolver` using model-guided scoring)
use `experiments/run_pipeline.py` - configurable in the script header.
Set either a single permutation (`STATE = [1,0,2]`) or read from a file (`STATE = None`, `FILE_PATH`).

```bash
uv run experiments/run_pipeline.py
```

## Project Structure

```
MLPermutationSolver/
├── src/
│   ├── data/           # Random walk data generation, BFS distances
│   ├── models/         # ML model implementations
│   └── solvers/        # Beam search solvers
├── experiments/        # Benchmarking and analysis scripts
├── pyproject.toml      # Project config and dependencies (uv)
├── uv.lock             # Locked dependency versions
└── README.md           # This file
```

## Experiments

The `experiments/` directory contains scripts for:

- **Benchmarking** random walk implementations
- **Optimizing** hyperparameters  
- **Profiling** model and solver performance
- **Validating** solution correctness

See [experiments/README_exp.md](experiments/README_exp.md) for detailed descriptions of each script and their parameters.

## Algorithms

### BFS Distances (src/data_gen/bfs_distances.py)

Numba-optimized BFS over the full permutation graph (X, L, R moves). Computes
exact distances from the identity to every reachable state. Memory-bound:
max n=13 on RTX 3090 with 24 GB VRAM / 124 GB RAM (~70 GiB for n=13).
Typical runtimes: n=12 ~2 min, n=13 ~35 min.

### Random Walk Data Generation (src/data_gen/random_walks.py)

All dataset generators simulate sequences of states (permutations) starting from
the identity permutation. They output (X, y), where X is a batch of states and
y is a step index used as a training target / proxy for "search depth".

1) first_visit_random_walks
Unconstrained random walks. Repeats are allowed. After sampling, each unique
state is assigned y = the earliest step at which the state appeared in the
whole sample ("first-visit time"). Therefore y is a function of state only,
and (state, y) pairs collapse to unique states. This generator is very fast, but
can waste a large fraction of the sampling budget on duplicates for larger state_size.

2) nbt_random_walks
Per-walk self-avoiding (non-backtracking) random walks. Each walk tracks the
set of states it has visited and forbids transitions to already visited states
within that walk. If a walk has no valid move, it terminates early. The same
state may appear at different steps across different walks, so y is not a pure
function of state. This generator is significantly slower due to per-walk
history checks.

3) random_walks_beam_nbt
Beam-style sampling with global non-backtracking in a sliding history window.
At each step, expand the current beam of states by all moves, discard states
present in the recent global hash history, then randomly select a new beam of
size n_walks from the remaining candidates. The target uses an "effective step"
(counter of successful expansions); when the state-space saturates, effective
step may stop increasing even if the loop continues. This variant typically
maximizes unique-state coverage per fixed sampling budget and is much faster
than per-walk self-avoidance.


### ML Models (src/models/)
The solver treats a permutation state as a feature vector (the raw sequence of
integers) and learns a scalar score y used as a proxy for "how deep / hard"
a state is in the search graph.

All models implement BaseModel with four methods:
- train(X, y): fit on a batch of states X and scalar targets y
- predict(X): return a 1D tensor of scores for states X
- save(path), load(path): persist and restore a trained model

1) XGBoostModel (src/models/xgboost_model.py)
Gradient-boosted trees trained and inferred fully on GPU. PyTorch CUDA tensors
are converted to CuPy via DLPack, XGBoost uses a GPU DMatrix for training,
and inference uses inplace_predict().
Predictions are returned as a CUDA torch.Tensor via DLPack zero-copy.

2) CatBoostModel (src/models/catboost_model.py)
CatBoostRegressor configured for GPU training. However, this wrapper currently
moves tensors to CPU (X.cpu().numpy(), y.cpu().numpy()) for both training and
prediction, then wraps predictions back into a torch tensor.

3) MLPModel (src/models/mlp_model.py)
A feed-forward neural network (Linear + BatchNorm + ReLU + Dropout blocks)
trained with Adam and MSE loss. This is a straightforward regression baseline
that runs on the same device as X. The checkpoint includes both model and
optimizer state, plus the architecture/training hyperparameters needed to
reconstruct the network on load.

### Beam Search Solvers (src/solvers/)
- ML-guided state evaluation
- Memory-efficient hash tracking
- GPU-accelerated batch processing
- Configurable pruning strategies

## Performance

The system efficiently handles:
- State sizes up to 34+ elements (on modern GPUs)
- GPU memory optimization

## Research

This work builds upon and contributes to research in AI-based approaches to Cayley graph pathfinding. Results using these methods have been reported in:

- Chervov, A., et al. (2025). "CayleyPy RL: Pathfinding and Reinforcement Learning on Cayley Graphs." [arXiv:2502.18663](https://arxiv.org/abs/2502.18663)

## License

This project is open source.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{mlpermutationsolver,
  title={MLPermutationSolver: Machine Learning Approach to Permutation Sorting},
  author={Dolgorukova, Antonina},
  year={2024},
  url={https://github.com/T0chka/MLPermutationSolver}
}

@article{chervov2025cayleypy,
  title={CayleyPy RL: Pathfinding and Reinforcement Learning on Cayley Graphs},
  author={Chervov, A. and Soibelman, A. and Lytkin, S. and others},
  journal={arXiv preprint arXiv:2502.18663},
  year={2025}
}
```

## Contact

For questions and collaboration:
- Email: an.dolgorukova@gmail.com
- GitHub: [@T0chka](https://github.com/T0chka) 
