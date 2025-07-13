# Feedback on Bayesian Inference Code

## Summary of Issues Found

### 1. **Bug in `log_likelihood` Function**
The main issue is in the likelihood calculation:

```python
# INCORRECT
def log_likelihood(samples, likelihood_mu, likelihood_sd):
    log_like, ll = [], []
    for data in samples:
        ll.append( -(data - likelihood_mu)**2 / (2*likelihood_sd**2))   
        log_like.append(np.sum(ll))  # BUG: ll keeps accumulating!
    return np.array(log_like)
```

**Problem**: The `ll` array is never reset, so each iteration includes all previous values. This causes the log-likelihood to become increasingly negative for later samples, creating the false impression that μ ≈ 0 has the highest posterior.

### 2. **Conceptual Confusion**
The code seems to mix up two different problems:
- **Parameter estimation**: Given observed data, what's the most likely value of μ?
- **Model comparison**: Given different models (different μ values), which generated the data?

### 3. **Incorrect Problem Setup**
The current approach generates different datasets for each μ value, then compares their likelihoods. This doesn't make sense for parameter estimation.

## Corrected Approach

### The Right Way to Think About It:
1. **Observe some data** (generated from unknown μ)
2. **Define prior beliefs** about μ (uniform distribution)
3. **Calculate likelihood** of observing this data for each possible μ value
4. **Compute posterior** using Bayes' rule: P(μ|data) ∝ P(data|μ) × P(μ)

### Key Corrections:
```python
# CORRECT approach
def log_likelihood(mu_vals, data, sigma):
    """Calculate log likelihood for each μ value given the SAME observed data"""
    log_like = np.zeros_like(mu_vals)
    for i, mu in enumerate(mu_vals):
        # Sum of log probabilities for all data points
        log_like[i] = np.sum(stats.norm.logpdf(data, mu, sigma))
    return log_like
```

## What the Results Should Show

With the corrected code:
- **Prior**: Uniform (flat) distribution
- **Likelihood**: Peaks around the sample mean of observed data
- **Posterior**: Combines prior and likelihood, peaks near sample mean
- **Effect of more data**: Posterior becomes more concentrated around true value

## Expected Behavior

The posterior should:
1. **Peak near the sample mean** (not at μ = 0)
2. **Become more concentrated** with more data points
3. **Approach the true value** as sample size increases
4. **Show proper uncertainty quantification**

## Next Steps for Student

1. **Fix the likelihood calculation** - use the same observed data for all μ values
2. **Understand the problem setup** - clarify whether doing parameter estimation or model comparison
3. **Verify results make intuitive sense** - posterior should peak near sample mean
4. **Explore the effect of sample size** - more data should reduce uncertainty

The corrected script `corrected_bayesian_example.py` demonstrates the corrected approach and shows what the results should look like.