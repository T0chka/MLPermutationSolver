# MLPermutationSolver

A machine learning approach to solving permutation sorting problems using beam search and neural guidance.

## Overview

This project implements a novel approach to solve permutation sorting problems by combining:
- **Random walk data generation** for training ML models
- **Beam search algorithms** guided by machine learning predictions
- **Multiple model architectures** (XGBoost, CatBoost, MLP)

The system can efficiently find optimal solutions for permutation sorting problems using L (left shift), R (right shift), and X (swap) operations.

## Features

- 🔄 **Multiple random walk algorithms** for data generation
- 🤖 **ML model integration** (XGBoost, CatBoost, Neural Networks)
- 🔍 **Beam search solver** with ML guidance
- 📊 **Comprehensive benchmarking tools**
- ⚡ **GPU acceleration** support
- 🧪 **Extensive experiment suite**

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
git clone https://github.com/YourUsername/MLPermutationSolver.git
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

Run experiments:
```bash
# Benchmark random walk algorithms
python experiments/benchmark_random_walks.py --state-size 16

# Optimize parameters
python experiments/optimize_random_walks.py --state-size 8 --test-all

# Profile model performance  
python experiments/profile_model.py --model-type xgboost

# Profile solver performance
python experiments/profile_solver.py --state-size 16
```

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
- State sizes up to 34 elements
- GPU memory optimization
- Sub-second solutions for many problems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please see the license file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{mlpermutationsolver,
  title={MLPermutationSolver: Machine Learning Approach to Permutation Sorting},
  author={Dolgorukova, Antonina},
  year={2024},
  url={https://github.com/YourUsername/MLPermutationSolver}
}
```

## Contact

For questions and collaboration:
- Email: an.dolgorukova@gmail.com
- GitHub: [@YourUsername](https://github.com/YourUsername) 