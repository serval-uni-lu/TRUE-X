# utils/seeding.py
import random, numpy as np, torch
def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
