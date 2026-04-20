import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def create_dirs():
    """Create necessary folders if they don't exist."""
    dirs = ['data/raw', 'models', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)