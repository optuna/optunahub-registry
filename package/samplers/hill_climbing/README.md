# Hill Climbing Sampler

## Abstract

The hill climbing algorithm is an optimization technique that iteratively improves a solution by evaluating neighboring solutions in search of a local maximum or minimum. Starting with an initial guess, the algorithm examines nearby "neighbor" solutions, moving to a better neighbor if one is found. This process continues until no improvement can be made locally, at which point the algorithm may restart from a new random position.

This implementation focuses on discrete optimization problems, supporting integer and categorical parameters only.

## Class or Function Names

- `HillClimbingSampler`

## Installation

No additional dependencies are required beyond Optuna and OptunaHub.

```bash
pip install optuna optunahub
```

## APIs

### HillClimbingSampler

```python
HillClimbingSampler(
    search_space: dict[str, BaseDistribution] | None = None,
    *,
    seed: int | None = None,
    neighbor_size: int = 5,
    max_restarts: int = 10,
)
```

#### Parameters

- **search_space** (`dict[str, BaseDistribution] | None`, optional): A dictionary containing the parameter names and their distributions. If `None`, the search space is inferred from the study.

- **seed** (`int | None`, optional): Seed for the random number generator to ensure reproducible results.

- **neighbor_size** (`int`, default=5): Number of neighboring solutions to generate and evaluate in each iteration.

- **max_restarts** (`int`, default=10): Maximum number of times the algorithm will restart from a random position when no improvements are found.

#### Supported Distributions

- `IntDistribution`: Integer parameters with specified bounds
- `CategoricalDistribution`: Categorical parameters with discrete choices

#### Limitations

- **Discrete only**: This sampler only supports discrete parameter types (`suggest_int` and `suggest_categorical`). Continuous parameters (`suggest_float`) are not supported.
- **Single-objective**: Only single-objective optimization is supported.

## Example

```python
import optuna
import optunahub

def objective(trial):
    # Integer parameter
    x = trial.suggest_int("x", -10, 10)
    
    # Categorical parameter  
    algorithm = trial.suggest_categorical("algorithm", ["A", "B", "C"])
    
    # Simple objective function
    penalty = {"A": 0, "B": 1, "C": 2}[algorithm]
    return x**2 + penalty

# Load the hill climbing sampler
module = optunahub.load_module("samplers/hill_climbing")
sampler = module.HillClimbingSampler(
    neighbor_size=8,    # Generate 8 neighbors per iteration
    max_restarts=5,     # Allow up to 5 restarts
    seed=42            # For reproducible results
)

# Create study and optimize
study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
```

## Algorithm Details

### Initialization

The algorithm starts with a random point in the parameter space using Optuna's `RandomSampler`.

### Neighbor Generation

For each iteration, the algorithm generates neighboring solutions by:

- **Integer parameters**: Adding or subtracting a small step size (based on the parameter range)
- **Categorical parameters**: Randomly selecting a different category from the available choices

### Movement Strategy

The algorithm evaluates all generated neighbors and moves to the best improvement found. If no improvement is discovered, it continues searching or restarts from a new random position.

### Restart Mechanism

When the algorithm gets stuck in a local optimum (no improvements found in recent iterations), it restarts from a new random position up to `max_restarts` times.

### State Management

The sampler maintains:

- Current best position and value
- List of neighbors to evaluate
- Set of already evaluated neighbors (to avoid re-evaluation)
- Restart counter

## Performance Characteristics

### Strengths

- **Simple and interpretable**: Easy to understand and debug
- **Good for discrete problems**: Well-suited for combinatorial optimization
- **Local exploitation**: Effective at refining solutions in promising regions
- **Memory efficient**: Low memory footprint compared to population-based algorithms

### Limitations

- **Local optima**: May get trapped in local optima without sufficient restarts
- **No global view**: Lacks global search capability of more sophisticated algorithms
- **Parameter sensitivity**: Performance depends on `neighbor_size` and `max_restarts` settings
- **Discrete only**: Cannot handle continuous parameters

## When to Use

This sampler is most effective for:

- Discrete/combinatorial optimization problems
- Problems with relatively small search spaces
- Situations where interpretability is important
- As a baseline for comparison with more complex algorithms
- Problems where local search is sufficient

## References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Chapter on Local Search Algorithms.
- Hoos, H. H., & Stützle, T. (2004). *Stochastic Local Search: Foundations and Applications*. Morgan Kaufmann.
