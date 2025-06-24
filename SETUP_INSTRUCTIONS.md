# Setup Instructions for SBI Feature Extraction Project

## Environment Setup

### 1. Python Environment
Create a dedicated conda environment for this project:

```bash
# Create new environment
conda create -n sbi-features python=3.9
conda activate sbi-features

# Or using mamba (faster)
mamba create -n sbi-features python=3.9
mamba activate sbi-features
```

### 2. Install Dependencies

#### Core Dependencies
```bash
# Install PyTorch (adjust for your system)
conda install pytorch torchvision torchaudio -c pytorch

# Install scientific computing stack
conda install numpy scipy matplotlib seaborn pandas scikit-learn jupyter

# Install SBI libraries
pip install sbi sbibm

# Install visualization tools
pip install corner plotly bokeh

# Install development tools
pip install tqdm pytest black flake8 isort
```

#### Install mini_falcon
```bash
# Navigate to the mini_falcon directory
cd ../naive-one-shot-sbi

# Install in development mode
pip install -e .

# Verify installation
python -c "import mini_falcon; print('mini_falcon installed successfully')"
```

### 3. Verify Installation
Run this test script to verify all dependencies:

```python
# test_installation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import sbi
import sbibm
import mini_falcon
import corner

print("All dependencies installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"SBI version: {sbi.__version__}")
```

## Project Structure Setup

### 1. Create Missing Directories
The main directories should already exist, but create any missing ones:

```bash
# Ensure all experiment subdirectories exist
mkdir -p experiments/{analysis,utils}
mkdir -p results/{figures,data,models}
mkdir -p docs
```

### 2. Initialize Git (if not already done)
```bash
git init
git add .
git commit -m "Initial project structure"
```

### 3. Create Configuration Files

#### Create `requirements.txt`
```txt
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
jupyter>=1.0.0
corner>=2.2.0
tqdm>=4.62.0
sbi>=0.19.0
sbibm>=1.0.0
plotly>=5.0.0
bokeh>=2.4.0
pytest>=6.0.0
black>=21.0.0
flake8>=3.9.0
isort>=5.9.0
```

#### Create `.gitignore`
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebooks
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Results and data (large files)
results/data/*.pkl
results/data/*.h5
results/models/*.pt
results/models/*.pth

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### 4. Create Initial Utility Files

#### Create `experiments/utils/__init__.py`
```python
"""
Utility functions for SBI feature extraction experiments.
"""

from .data_utils import *
from .training_utils import *
from .analysis_utils import *
from .visualization_utils import *
```

#### Create basic utility file templates
```bash
# Create empty utility files for student to implement
touch experiments/utils/data_utils.py
touch experiments/utils/training_utils.py  
touch experiments/utils/analysis_utils.py
touch experiments/utils/visualization_utils.py
```

## Development Workflow

### 1. Daily Workflow
```bash
# Start of day
conda activate sbi-features
cd /path/to/sbi-features
git pull  # if working with remote repo

# During development
jupyter lab  # or jupyter notebook

# End of day
git add .
git commit -m "Descriptive commit message"
git push  # if using remote repo
```

### 2. Running Experiments
```python
# Template for running experiments
import sys
sys.path.append('../naive-one-shot-sbi')
import mini_falcon as mfalcon

# Your experiment code here
```

### 3. Code Organization
- Keep notebooks in `notebooks/week_X/` directories
- Extract reusable functions to `experiments/utils/`
- Save experimental results to `results/` subdirectories
- Document everything clearly

## Testing Your Setup

### 1. Test Basic SBI Functionality
```python
# test_sbi_basic.py
import torch
import sbi
import sbibm

# Test sbibm benchmark
task = sbibm.get_task("two_moons")
prior = task.get_prior_dist()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)

print("SBI basic functionality working!")
```

### 2. Test mini_falcon Integration
```python
# test_mini_falcon.py
import sys
sys.path.append('../naive-one-shot-sbi')
import mini_falcon as mfalcon
import torch

# Test basic functionality
print("mini_falcon imported successfully!")

# Test dataset creation (should work if mini_falcon is properly installed)
try:
    import sbibm
    task = sbibm.get_task("two_moons")
    prior = task.get_prior_dist()
    simulator = task.get_simulator()
    
    dataset = mfalcon.MiniFalconDataset(
        num_samples=100,
        prior=prior,
        simulator=simulator,
    )
    print("mini_falcon dataset creation working!")
except Exception as e:
    print(f"Error: {e}")
    print("Check mini_falcon installation")
```

### 3. Test Jupyter Environment
```bash
# Start Jupyter and test basic functionality
jupyter lab

# In a new notebook, test:
# - Import all required libraries
# - Create basic plots
# - Run simple SBI example
```

## Troubleshooting Common Issues

### 1. Import Errors
**Problem**: `ImportError: No module named 'mini_falcon'`
**Solution**: 
```bash
cd ../naive-one-shot-sbi
pip install -e .
```

### 2. PyTorch Installation Issues
**Problem**: PyTorch not working with GPU/different versions
**Solution**: Visit https://pytorch.org/get-started/locally/ and install appropriate version

### 3. Jupyter Not Finding Environment
**Problem**: Jupyter doesn't see conda environment
**Solution**:
```bash
conda activate sbi-features
python -m ipykernel install --user --name sbi-features --display-name "SBI Features"
```

### 4. Permission Errors
**Problem**: Permission denied when installing packages
**Solution**: Make sure you're in the correct conda environment, don't use sudo with conda

### 5. Package Version Conflicts
**Problem**: Dependency conflicts between packages
**Solution**: Create fresh environment and install minimal required packages first

## Next Steps

After completing setup:

1. **Week 1**: Start with `notebooks/week_1/` and `LEARNING_RESOURCES.md`
2. **Read Documentation**: Review all instruction files in experiment directories
3. **Test Examples**: Run examples from `../naive-one-shot-sbi/examples/`
4. **Begin Literature Review**: Start collecting and reading key papers

## Getting Help

### Resources
- **SBI Documentation**: https://sbi-dev.github.io/sbi/
- **sbibm Documentation**: https://sbibm.readthedocs.io/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Project Issues**: Document problems in notebook markdown cells

### Contact
- **Supervisor**: James Alvey (for research questions)
- **Technical Issues**: Document in project notebooks for discussion

Remember: The goal is learning, so don't hesitate to experiment and ask questions!