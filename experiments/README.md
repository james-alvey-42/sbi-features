# Experiments Directory

This directory contains the experimental code for studying summary statistics learned by SBI algorithms.

## Directory Structure

### `known_optimal/`
Experiments with distributions where optimal summary statistics are analytically known.
- Gaussian distributions (mean, variance, covariance estimation)
- Exponential family distributions (Poisson, exponential, beta)
- Goal: Validate that SBI learns embeddings equivalent to known optimal summaries

### `intermediate/`
Experiments with moderate complexity where optimal summaries are less clear.
- Gaussian mixture models
- Simple dynamical systems (SLCP)
- Goal: Develop techniques for analyzing learned summaries

### `complex/`
Experiments with benchmark problems and real-world applications.
- Two moons (sbibm benchmark)
- Gravitational wave toy problems
- Goal: Apply explainable AI techniques to understand learned compressions

### `analysis/`
Shared analysis tools and metrics for comparing embeddings.
- Distance metrics between learned and optimal summaries
- Visualization tools for embedding spaces
- Statistical testing frameworks

### `utils/`
Utility functions for data generation, model training, and evaluation.
- Data simulation helpers
- Training loop implementations
- Evaluation metrics

## Getting Started

1. Start with experiments in `known_optimal/` to validate the approach
2. Use tools from `analysis/` and `utils/` to build your analysis pipeline
3. Progress to `intermediate/` and `complex/` experiments as understanding develops

## Deliverables

Each experiment should produce:
- Jupyter notebook with analysis and visualizations
- Python module with reusable functions
- Results saved to `../results/` directory
- Written documentation of findings and interpretations (e.g. in a .md file)