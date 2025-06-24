# Experimental Design for SBI Summary Statistics Analysis

## Overview

This document outlines the experimental framework for systematically studying the summary statistics learned by simulation-based inference algorithms, progressing from known optimal cases to complex scenarios requiring explainable AI techniques.

## Experimental Methodology

### Core Research Questions
1. **Validation Question**: Do SBI algorithms learn embeddings equivalent to known optimal summary statistics?
2. **Discovery Question**: What summary statistics are learned when optimal summaries are unknown?
3. **Interpretability Question**: How can we develop explainable AI frameworks for understanding learned compressions?

### General Experimental Protocol
1. **Problem Setup**: Define simulator, prior, and observation
2. **Baseline Creation**: Implement known optimal summaries (when available)
3. **SBI Training**: Train embedding networks with different architectures
4. **Comparison Analysis**: Quantitative and qualitative comparison of learned vs. optimal summaries
5. **Visualization**: Create interpretable visualizations of embedding spaces
6. **Statistical Testing**: Perform hypothesis tests on summary equivalence

## Experiment Categories

### Category A: Known Optimal Cases (Validation)

#### Experiment A1: Univariate Gaussian Mean Estimation
**Setup:**
- Simulator: `x ~ N(θ, σ²)` with known σ
- Prior: `θ ~ N(μ₀, τ²)`
- Optimal summary: `T(x) = x̄` (sample mean)

**Example Implementation:**
```python
def gaussian_mean_experiment():
    # Known optimal summary
    def optimal_summary(x):
        return x.mean(dim=-1, keepdim=True)
    
    # Train SBI with different embedding dimensions
    embedding_dims = [1, 2, 4, 8]
    results = {}
    
    for dim in embedding_dims:
        embedding_net = create_embedding_net(input_dim=n_obs, output_dim=dim)
        trained_model = train_sbi(embedding_net, ...)
        results[dim] = analyze_embeddings(trained_model, optimal_summary)
    
    return results
```

**Analysis Goals:**
- Verify 1D embedding learns sample mean
- Study behavior with over-parameterized embeddings
- Measure distance between learned and optimal summaries

#### Experiment A2: Multivariate Gaussian Covariance
**Setup:**
- Simulator: `x ~ N(μ, Σ)` with known μ
- Prior: `Σ ~ InverseWishart(ψ, ν)`
- Optimal summary: `T(x) = (x-μ)(x-μ)ᵀ` (sample covariance)

**Key Metrics:**
- Embedding dimension sensitivity

#### Experiment A3: Exponential Family Distributions
**Setup Multiple Distributions:**
- Poisson: `x ~ Poisson(λ)`, optimal summary `T(x) = Σx`
- Exponential: `x ~ Exp(λ)`, optimal summary `T(x) = Σx`
- Beta: `x ~ Beta(α, β)`, optimal summaries `T(x) = [Σlog(x), Σlog(1-x)]`

### Category B: Intermediate Complexity Cases

#### Experiment B1: Gaussian Mixture Models
**Setup:**
- Simulator: `x ~ πN(μ₁, σ₁²) + (1-π)N(μ₂, σ₂²)`
- Unknown optimal summaries
- Parameters: `θ = [π, μ₁, μ₂, σ₁, σ₂]`

**Analysis Framework:**
- Compare learned embeddings across different mixture complexities
- Study interpretations of the embeddings
- Study dimensionality dependence of embedding

#### Experiment B2: Lotka-Volterra
**Setup:**
- Dynamical system with known ground truth
- Time series observations
- Parameters: growth rates, interaction strengths

### Category C: Complex Benchmark Problems

#### Experiment C1: Two Moons (sbibm)
**Setup:**
- Standard sbibm benchmark
- Non-linear decision boundaries
- Well-characterized posterior

**Advanced Analysis:**
- Interpretation of embeddings

#### Experiment C2: Gravitational Wave Toy Problem
**Setup:**
- Simplified GW waveform simulation
- Noise modeling
- Physical parameter estimation

**Research Value:**
- Bridge to real astrophysics applications
- High-dimensional signal processing
- Domain-specific summary statistics

## Reproducibility Standards

### Code Organization
```
experiments/
├── known_optimal/
│   ├── gaussian_mean.py
│   ├── gaussian_covariance.py
│   └── exponential_family.py
├── intermediate/
│   ├── gaussian_mixtures.py
│   └── slcp.py
├── complex/
│   ├── two_moons.py
│   └── gravitational_waves.py
├── analysis/
│   ├── metrics.py
│   ├── visualization.py
│   └── statistical_tests.py
└── utils/
    ├── data_generation.py
    ├── model_training.py
    └── evaluation.py
```

### Documentation Requirements
- Experiment configuration files (YAML/JSON)
- Detailed docstrings for all functions
- Version control for all experiments
- Automated testing for analysis functions
- Results logging and experiment tracking

### Qualitative Assessments
- Clear interpretability of learned features
- Meaningful visualizations of embedding spaces
- Coherent narrative connecting theory to results
- Actionable insights for future research