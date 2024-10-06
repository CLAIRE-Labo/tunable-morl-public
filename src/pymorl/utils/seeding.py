import random

import numpy as np
import torch


def generate_random_seed():
    """Generate a random seed."""
    return random.randint(0, 2**32 - 1)


def seed_everything(config):
    """Seed all random generators."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
