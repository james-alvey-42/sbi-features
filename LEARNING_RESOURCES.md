# Learning Resources for SBI Feature Extraction Project

## Phase 1: Bayesian Inference Fundamentals

### Essential Concepts to Master
- **Bayes' Theorem**: P(θ|x) = P(x|θ)P(θ)/P(x)
- **Prior Distribution**: Encodes initial beliefs about parameters
- **Likelihood Function**: Probability of observing data given parameters
- **Posterior Distribution**: Updated beliefs after observing data
- **Marginal Likelihood**: Evidence or normalization constant

### Recommended Readings
1. **"Bayesian Data Analysis" (Gelman et al.)** - Chapters 1-2
2. **"Information Theory, Inference, and Learning Algorithms" (MacKay)** - Chapters 1-3

### Key Questions to Answer
- When is analytical Bayesian inference possible?
- What makes a problem "intractable" for traditional methods?
- How do sufficient statistics reduce computational complexity?

## Phase 2: Summary Statistics and Information Theory

### Core Concepts
- **Sufficient Statistics**: Capture all information about parameters
- **Fisher Information**: Measure of information content
- **Mutual Information**: I(X;Y) - measures dependence between variables
- **Data Compression**: Lossy vs. lossless compression principles

### Mathematical Foundations
- **Fisher-Neyman Factorization Theorem**
- **Kullback-Leibler Divergence**: D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
- **Information Bottleneck Principle**

## Phase 3: Simulation-Based Inference

### Technical Understanding Goals
- Why traditional methods can fail
- Amortized vs. sequential inference
- SBI Algorithms: Neural density estimation vs. ratio estimation
- Embedding networks and their role

## Phase 4: Python & Machine Learning Essentials

### Python Libraries to Master
```python
# Core scientific computing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Machine learning
import torch
import torch.nn as nn
import sklearn

# SBI specific
import sbi
import sbibm
import mini_falcon

# Visualization
import corner
import seaborn as sns
```

### PyTorch Fundamentals
- Tensor operations and automatic differentiation
- Building neural networks with `nn.Module`
- Training loops and optimization
- GPU acceleration basics

### Key Skills to Develop
- Data visualization with matplotlib/seaborn
- Statistical analysis with NumPy/SciPy
- Neural network debugging and monitoring
- Git version control for research

## Hands-On Learning Exercises

### Week 1 Exercises
1. **Analytical Bayesian Inference**
   - Implement prior and likelihood distributions by hand, and compute the posterior
   - Implement simple MCMC examples to find posterior results with stochastic sampling
   - Visualize prior, likelihood, and posteriors for simple problems
   
2. **Summary Statistics Practice**
   - Understand sufficient statistics for exponential family of distributions
   - Implement Fisher Information calculation

### Week 2 Exercises
1. **SBI Library Tutorial**
   - Follow official SBI documentation examples
   - Run inference on simple problems (Gaussian, two_moons)
   
2. **mini_falcon Exploration**
   - Study provided examples in naive-one-shot-sbi/examples/
   - Modify embedding network architectures
   - Compare training dynamics

### Week 3-4 Project Exercises (to be extended)
- Implement from scratch: Gaussian mean estimation
- Verify learned embeddings match analytical solutions
- Design custom visualization tools
- Develop quantitative comparison metrics

## Assessment Checkpoints

### Week 1 Self-Assessment
- [ ] Can explain Bayes' theorem with concrete examples
- [ ] Understands role of priors in inference
- [ ] Can implement simple inference examples
- [ ] Comfortable with basic Python scientific computing

### Week 2 Self-Assessment
- [ ] Understands when SBI could be used
- [ ] Can run basic SBI examples successfully
- [ ] Know how to modify embedding networks
- [ ] Can interpret posterior samples and make corner plots

### Week 3 Self-Assessment
- [ ] Successfully implemented known optimal cases
- [ ] Can compare learned vs. analytical summaries quantitatively
- [ ] Developed working visualization tools
- [ ] Understands experimental design principles

### Week 4 Self-Assessment
- [ ] Analyzed complex benchmark problems
- [ ] Built comprehensive analysis toolkit for exploring learned summaries
- [ ] Can articulate findings and limitations
- [ ] Prepared future directions document

## Additional Resources

### Software Documentation
- [SBI Documentation](https://sbi-dev.github.io/sbi/)
- [sbibm Benchmarks](https://sbibm.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Troubleshooting Common Issues

### Coding Challenges
- **Environment Setup**: Use conda/mamba for dependency management
- **GPU Issues**: Start with CPU, add GPU acceleration later

### Conceptual Challenges
- **Information Theory**: Focus on intuitive understanding before mathematical rigor
- **Neural Networks**: Start with simple architectures, add complexity gradually
- **SBI Theory**: Connect back to traditional Bayesian inference frequently 