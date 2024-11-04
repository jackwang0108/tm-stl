# Standard Library
import os
import random
from pathlib import Path

# Third-Party Library
import numpy as np

# Torch Library
import torch


def set_random_seed(seed: int) -> None:
    """设置多个库的随机种子以确保结果可重复。

    该函数配置Python内置随机模块、NumPy和PyTorch的随机种子，包括CPU和GPU设置。通过设置种子，可以帮助在代码的不同运行之间产生一致的结果。

    Args:
        seed (int): 要设置的随机数生成种子值。

    Returns:
        None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
