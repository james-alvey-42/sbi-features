# Notebooks Directory

This directory contains Jupyter notebooks organized by project week, documenting the learning process and experimental results.

## Directory Structure

### `week_1/`
Learning phase notebooks covering foundational concepts:
- Bayesian inference basics and examples
- Python and machine learning fundamentals  
- Literature review summaries
- Initial SBI library exploration

### `week_2/`
SBI foundations and practical implementation:
- Simulation-based inference theory
- mini_falcon framework tutorials
- First experimental implementations
- Summary statistics deep dive

### `week_3/`
Known optimal cases implementation and analysis:
- Gaussian distribution experiments
- Exponential family validations
- Quantitative comparison methods
- Embedding visualization techniques

### `week_4/`
Complex cases and explainable AI development:
- Intermediate complexity experiments
- Benchmark problem analysis
- Explainability framework development
- Final project synthesis

## Notebook Guidelines

### Naming Convention
Use descriptive names with date prefixes:
- `YYYY-MM-DD_topic_description.ipynb`
- Example: `2024-07-01_gaussian_mean_validation.ipynb`

### Structure Template
Each notebook should include:

```markdown
# Notebook Title
**Date:** YYYY-MM-DD  
**Objective:** Clear statement of what this notebook accomplishes  
**Key Results:** Summary of main findings

## Background
Brief context and motivation

## Methods
Implementation approach and key functions

## Results
Analysis with visualizations and interpretations

## Conclusions
Key takeaways and next steps

## References
Relevant papers, documentation, or resources
```

### Best Practices
- Include clear markdown explanations between code cells
- Save key figures to `../results/figures/` directory
- Export important data/results to `../results/data/`
- Use version control to track notebook evolution
- Include reproducibility information (random seeds, package versions)

## Collaboration and Sharing

### Version Control
- Commit notebooks regularly with meaningful messages
- Use `.gitignore` to exclude large output files
- Consider using nbstripout to clean notebooks before committing

### Documentation
- Include detailed docstrings in notebook functions
- Export reusable code to modules in `../experiments/utils/`
- Maintain a research log of key decisions and insights

### Reproducibility
- Use `requirements.txt` or `environment.yml` for dependencies
- Set random seeds for reproducible results
- Document computational environment details
- Save intermediate results for complex computations
