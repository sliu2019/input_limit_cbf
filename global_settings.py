import torch
import numpy as np

# TODO: this makes torch and numpy deterministic.
# TODO: This may not be what you want for, (for example) multiple identical runs with different randomness
torch.manual_seed(10)
np.random.seed(2021)
