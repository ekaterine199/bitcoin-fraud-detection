import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    # 1. Basic Python and OS seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you use multi-GPU
    
    # 3. Deterministic algorithms for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"✅ Deterministic mode enabled with seed: {seed}")