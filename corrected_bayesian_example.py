"""
Corrected Bayesian Inference Example
====================================

This script demonstrates proper Bayesian inference with:
- Uniform prior on parameter μ
- Gaussian likelihood for observed data
- Analytical posterior calculation

The key insight: We observe some data and want to infer the parameter μ
that generated it, given our prior beliefs about μ.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Problem setup
# =============
# We have a parameter μ that we want to estimate
# Prior: μ ~ Uniform(0, 2)  [uniform prior on μ]
# Likelihood: X|μ ~ Normal(μ, σ²)  [data comes from normal distribution with mean μ]
# We observe some data points and want to find posterior P(μ|data)

# True parameter (unknown in practice)
true_mu = 1.2
sigma = 0.3  # Known standard deviation (for simplicity)

# Generate observed data
n_observations = 10
observed_data = np.random.normal(true_mu, sigma, n_observations)

print(f"True μ: {true_mu}")
print(f"Observed data: {observed_data}")
print(f"Sample mean: {np.mean(observed_data):.3f}")
print(f"Sample std: {np.std(observed_data):.3f}")

# Define parameter space
mu_values = np.linspace(-0.5, 3.0, 1000)

# Prior P(μ)
# ==========
# Uniform prior between 0 and 2
prior_low, prior_high = 0.0, 2.0
prior = np.where((mu_values >= prior_low) & (mu_values <= prior_high), 
                 1.0/(prior_high - prior_low), 0.0)

# Likelihood P(data|μ)
# ===================
# For each value of μ, calculate likelihood of observing our data
# L(μ) = ∏ N(x_i | μ, σ²)
# log L(μ) = Σ log N(x_i | μ, σ²)

def log_likelihood(mu_vals, data, sigma):
    """Calculate log likelihood for each μ value"""
    log_like = np.zeros_like(mu_vals)
    for i, mu in enumerate(mu_vals):
        # Sum of log probabilities for all data points
        log_like[i] = np.sum(stats.norm.logpdf(data, mu, sigma))
    return log_like

log_likelihood_vals = log_likelihood(mu_values, observed_data, sigma)
likelihood_vals = np.exp(log_likelihood_vals)

# Posterior P(μ|data) ∝ P(data|μ) × P(μ)
# =====================================
# Unnormalized posterior
posterior_unnorm = likelihood_vals * prior

# Normalize posterior (integrate to 1)
posterior = posterior_unnorm / np.trapz(posterior_unnorm, mu_values)

# Analytical solution for comparison
# ==================================
# For uniform prior + normal likelihood, the posterior is also normal
# Posterior mean: (n*x_bar*σ_prior² + μ_prior*σ²) / (n*σ_prior² + σ²)
# But with uniform prior, it simplifies to just the sample mean in the limit

# For uniform prior on [a,b] and normal likelihood:
# If sample mean is within [a,b], posterior ≈ Normal(sample_mean, σ²/n)
sample_mean = np.mean(observed_data)
posterior_std = sigma / np.sqrt(n_observations)

analytical_posterior = stats.norm.pdf(mu_values, sample_mean, posterior_std)
# Truncate to prior bounds
analytical_posterior = np.where((mu_values >= prior_low) & (mu_values <= prior_high), 
                               analytical_posterior, 0.0)
analytical_posterior = analytical_posterior / np.trapz(analytical_posterior, mu_values)

# Summary statistics
# ==================
# Find MAP (Maximum A Posteriori) estimate
map_estimate = mu_values[np.argmax(posterior)]

# Calculate posterior mean and credible interval
posterior_mean = np.trapz(mu_values * posterior, mu_values)
posterior_var = np.trapz((mu_values - posterior_mean)**2 * posterior, mu_values)
posterior_std_computed = np.sqrt(posterior_var)

print(f"\nResults:")
print(f"MAP estimate: {map_estimate:.3f}")
print(f"Posterior mean: {posterior_mean:.3f}")
print(f"Posterior std: {posterior_std_computed:.3f}")
print(f"95% credible interval: [{posterior_mean - 1.96*posterior_std_computed:.3f}, {posterior_mean + 1.96*posterior_std_computed:.3f}]")

# Plotting
# ========
plt.figure(figsize=(12, 10))

# Plot 1: Prior, Likelihood, and Posterior
plt.subplot(2, 2, 1)
plt.plot(mu_values, prior, 'b-', label='Prior P(μ)', linewidth=2)
plt.plot(mu_values, likelihood_vals/np.max(likelihood_vals), 'r--', label='Likelihood (normalized)', linewidth=2)
plt.plot(mu_values, posterior, 'g-', label='Posterior P(μ|data)', linewidth=2)
plt.axvline(true_mu, color='black', linestyle=':', alpha=0.7, label='True μ')
plt.axvline(sample_mean, color='orange', linestyle=':', alpha=0.7, label='Sample mean')
plt.axvline(map_estimate, color='green', linestyle=':', alpha=0.7, label='MAP estimate')
plt.xlabel('μ')
plt.ylabel('Density')
plt.title('Prior, Likelihood, and Posterior')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Log-likelihood
plt.subplot(2, 2, 2)
plt.plot(mu_values, log_likelihood_vals, 'r-', linewidth=2)
plt.axvline(true_mu, color='black', linestyle=':', alpha=0.7, label='True μ')
plt.axvline(sample_mean, color='orange', linestyle=':', alpha=0.7, label='Sample mean')
plt.xlabel('μ')
plt.ylabel('Log-likelihood')
plt.title('Log-likelihood P(data|μ)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Comparison with analytical solution
plt.subplot(2, 2, 3)
plt.plot(mu_values, posterior, 'g-', label='Numerical posterior', linewidth=2)
plt.plot(mu_values, analytical_posterior, 'g--', label='Analytical approximation', linewidth=2, alpha=0.7)
plt.axvline(true_mu, color='black', linestyle=':', alpha=0.7, label='True μ')
plt.xlabel('μ')
plt.ylabel('Density')
plt.title('Posterior Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Data and predictive distribution
plt.subplot(2, 2, 4)
# Plot observed data
plt.hist(observed_data, bins=8, density=True, alpha=0.6, color='skyblue', label='Observed data')
# Plot predictive distribution using posterior mean
x_pred = np.linspace(observed_data.min()-0.5, observed_data.max()+0.5, 100)
pred_dist = stats.norm.pdf(x_pred, posterior_mean, sigma)
plt.plot(x_pred, pred_dist, 'g-', linewidth=2, label=f'Predictive (μ={posterior_mean:.2f})')
# Plot true distribution
true_dist = stats.norm.pdf(x_pred, true_mu, sigma)
plt.plot(x_pred, true_dist, 'k--', linewidth=2, label=f'True (μ={true_mu:.2f})')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Data and Predictive Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional example: Effect of different amounts of data
# ======================================================
plt.figure(figsize=(12, 8))

n_data_points = [1, 2, 5, 10, 20, 50]
colors = plt.cm.viridis(np.linspace(0, 1, len(n_data_points)))

for i, n in enumerate(n_data_points):
    # Generate data
    data_subset = observed_data[:n] if n <= len(observed_data) else np.random.normal(true_mu, sigma, n)
    
    # Calculate likelihood and posterior
    log_like = log_likelihood(mu_values, data_subset, sigma)
    like = np.exp(log_like)
    post = like * prior
    post = post / np.trapz(post, mu_values)
    
    plt.subplot(2, 3, i+1)
    plt.plot(mu_values, prior, 'b-', alpha=0.5, label='Prior')
    plt.plot(mu_values, post, color=colors[i], linewidth=2, label=f'Posterior (n={n})')
    plt.axvline(true_mu, color='black', linestyle=':', alpha=0.7)
    plt.axvline(np.mean(data_subset), color='red', linestyle=':', alpha=0.7)
    plt.xlabel('μ')
    plt.ylabel('Density')
    plt.title(f'n = {n}, sample mean = {np.mean(data_subset):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.suptitle('Effect of Sample Size on Posterior', fontsize=16)
plt.tight_layout()
plt.show()

print(f"\nKey takeaways:")
print(f"1. The posterior peak moves toward the sample mean as we get more data")
print(f"2. The posterior becomes more concentrated (less uncertain) with more data")
print(f"3. The uniform prior has minimal effect when we have sufficient data")
print(f"4. The MAP estimate ≈ sample mean when the sample mean is within the prior bounds")