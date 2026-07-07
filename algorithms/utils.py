import random
import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Set RNG seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
