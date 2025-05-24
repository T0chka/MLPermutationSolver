# MLPermutationSolver

MLPermutationSolver is a GPU-accelerated PyTorch implementation of machine learning-guided beam search for solving permutation sorting problems on Cayley graphs with LRX generators (L - left shift, R - right shift, and X - swap of the first 2 elements).

This repo is a spinoff from CayleyPy by Alexander Chervov and collaborators (applying artificial intelligence methods to the mathematical problem of Cayley graph pathfinding). It represents a parallel implementation of the Beam seach results of testing which were reported in academic publications e.g. [here](https://arxiv.org/abs/2502.18663).

## Key Features

- **Random walk data generation algorithms** for training ML models
- **Beam search algorithms** guided by machine learning predictions
- **Multiple model architectures** (XGBoost, CatBoost, MLP)
- **GPU acceleration** (batching, inference, hashing)
- **Benchmarking, profiling, and evaluation tools**

## Installation

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib

### Setup
```bash
git clone https://github.com/T0chka/MLPermutationSolver.git
cd MLPermutationSolver
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from src import BeamSearchSolver, XGBoostModel, create_lrx_moves, random_walks_beam_nbt

# Generate training data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generators = create_lrx_moves(8)  # For 8-element permutations
X, y = random_walks_beam_nbt(generators, n_steps=28, n_walks=10000, device=device)

# Train model
model = XGBoostModel()
model.train(X, y)

# Solve a permutation
start_state = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], device=device)
solver = BeamSearchSolver(state_size=8, beam_width=16, max_steps=100, device=device)
found, steps, solution = solver.solve(start_state, model)

print(f"Solution found: {found}")
print(f"Steps: {steps}")
print(f"Moves: {solution}")
```

## Project Structure

```
MLPermutationSolver/
├── src/
│   ├── data/           # Random walk data generation
│   ├── models/         # ML model implementations  
│   └── solvers/        # Beam search solvers
├── experiments/        # Benchmarking and analysis scripts
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Experiments

The `experiments/` directory contains scripts for:

- **Benchmarking** random walk implementations
- **Optimizing** hyperparameters  
- **Profiling** model and solver performance
- **Validating** solution correctness

See [experiments/README.md](experiments/README.md) for detailed descriptions of each script and their parameters.

## Algorithms

### Random Walk Data Generation
- **First Visit**: Tracks when states are first encountered
- **Non-Backtracking (NBT)**: Avoids revisiting recent states
- **Beam NBT**: Efficient batched non-backtracking walks

### ML Models
- **XGBoost**: Gradient boosting for tabular data
- **CatBoost**: Catboost with GPU support
- **MLP**: Multi-layer perceptron with PyTorch

### Beam Search Solver
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