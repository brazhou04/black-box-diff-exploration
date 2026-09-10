from __future__ import annotations

import os
import random
from typing import Any


def set_all_seeds(seed: int, deterministic: bool = False) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded = {"python": seed, "dataset_shuffle": seed, "trainer": seed}
    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = seed
    except ImportError:
        seeded["numpy"] = None
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        seeded.update(
            pytorch=seed,
            cuda=seed if torch.cuda.is_available() else None,
            deterministic_algorithms=deterministic,
        )
    except ImportError:
        seeded.update(pytorch=None, cuda=None, deterministic_algorithms=False)
    return seeded

