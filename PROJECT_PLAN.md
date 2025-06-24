# SBI Feature Extraction Project Plan

**Student:** Susie Lei (SL)  
**Supervisor:** James Alvey (JA)  
**Duration:** 4 weeks  

## Project Overview

This project explores the types and classes of summary statistics learned by simulation-based inference (SBI) algorithms when trained on various statistical problems. The core research question is: What optimal compressions of input data do neural networks learn during simulation-based inference, and how do these compare to known optimal summaries?

## Learning Phase (Weeks 1-2)

### Week 1: Foundations
- **Bayesian Inference Basics**
  - Understanding likelihood, prior, and posterior
  - Bayes' theorem and its applications
  - Traditional inference methods vs. simulation-based approaches

- **Python & Machine Learning Fundamentals** (if needed)
  - NumPy, PyTorch basics
  - Neural network fundamentals
  - Data handling and visualization

- **Literature Review Focus Areas**
  - Summary statistics in statistical inference
  - Information theory and optimal compression
  - Neural posterior estimation (NPE) theory

### Week 2: SBI Foundations
- **Simulation-Based Inference Theory**
  - When might traditional methods fail
  - Simulation-baesd inference concepts and algorithms
  - Neural density estimation

- **Practical Implementation**
  - `sbi` library tutorial
  - `mini_falcon` framework understanding
  - Running first examples (two_moons, simple Gaussian)

- **Summary Statistics Deep Dive**
  - Role of summaries in reducing dimensionality
  - Sufficient statistics theory
  - Information-theoretic optimality

## Implementation Phase (Weeks 3-4)

### Week 3: Known Optimal Cases
**Objective:** Validate that SBI learns equivalent summaries to known optimal ones

- **Gaussian Examples**
  - Simple univariate Gaussian (model for mean/variance parameter estimation)
  - Multivariate Gaussian (covariance structure)
  - Compare learned embeddings to analytical optimal summaries

- **Exponential Family Distributions**
  - Poisson distribution (rate parameter)
  - Exponential distribution (scale parameter)
  - Verify embedding networks learn sufficient statistics

### Week 4: Unknown Optimal Cases & Analysis
**Objective:** Develop techniques for interrogating learned compressions

- **More Complex Benchmark Problems**
  - Two moons (sbibm benchmark)
  - Gaussian mixtures
  - Simple Likelihood, Complex Posterior (SLCP)
  - Lotka-Volterra
  - Gravitational wave example

- **Explainable AI Framework Development**
  - Feature visualization techniques
  - Embedding space analysis
  - Comparison tools for trained vs. random networks

## Deliverables

### Learning Phase Deliverables
1. **Literature Review Summary** (1-2 pages)
   - Key papers on SBI and summary statistics
   - Theoretical foundations documented

2. **Technical Setup Documentation**
   - Environment setup guide
   - First successful runs of example problems

### Implementation Phase Deliverables
1. **Known Optimal Cases Analysis**
   - Jupyter notebooks with Gaussian and exponential family experiments
   - Quantitative comparison of learned vs. optimal summaries
   - Visualization of embedding spaces

2. **Explainable AI Toolkit**
   - Functions for analyzing learned embeddings
   - Visualization tools for feature comparison
   - Documentation of analysis techniques

3. **Final Report & Presentation**
   - Summary of findings
   - Comparison of learned vs. optimal summaries
   - Recommendations for future work

## Testing Framework for Summary Statistics

### Possible Comparison Metrics
- **Distance Measure Ideas**
  - Correlation analysis between learned and optimal summaries
  - Mutual information between embeddings and parameters
  - KL divergence between posterior approximations

### Visualization Tools
- **Embedding Space Analysis**
  - t-SNE/UMAP of learned features
  - Corner plots of posterior samples
  - Feature importance heatmaps

### Experimental Design
- **Controlled Comparisons**
  - Trained vs. random embedding networks
  - Different embedding dimensions
  - Various training dataset sizes

## Progressive Complexity

### Phase 1: Simple Cases (Known Solutions)
- 1D Gaussian (mean estimation)
- 1D Exponential (rate estimation)
- 2D Gaussian (mean vector estimation)

### Phase 2: Intermediate Cases
- Multivariate Gaussian (full covariance)
- Gaussian mixtures (2-3 components)
- Poisson processes

### Phase 3: Complex Benchmarks
- Two moons (non-linear decision boundary)
- SLCP (dynamical system)
- Higher-dimensional problems

### Phase 4: Research Extensions (if time permits)
- Gravitational wave toy problems
- Custom summary statistics design
- Advanced explainability techniques

## Weekly Check-ins

### Week 1 Check-in
- Review literature progress
- Assess coding skill level
- Adjust learning materials if needed

### Week 2 Check-in
- Validate SBI understanding
- Review first implementation attempts
- Plan Week 3 experiments

### Week 3 Check-in
- Review known optimal case results
- Discuss findings and interpretations
- Refine Week 4 objectives

### Week 4 Check-in
- Review complex case analyses
- Prepare final report
- Document future research directions

## Resources

### Key Papers to Review
- "The frontier of simulation-based inference" (Cranmer et al.)

### Technical Resources
- `sbi` library documentation
- `sbibm` benchmark suite
- `mini_falcon` examples in naive-one-shot-sbi/examples/

### Supervisor Support
- Weekly 1-hour meetings with JA
- As-needed short check-ins during implementation phases
- Code review and feedback on Github