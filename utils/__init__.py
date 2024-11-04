# Standard Library
import os
import sys
import random
from pathlib import Path

# Third-Party Library
import numpy as np
from loguru import logger

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


def get_device():
    """确定用于张量计算的合适设备。

    该函数检查CUDA和金属性能着色器（MPS）的可用性，以选择最适合
    运行PyTorch操作的设备。它返回一个PyTorch设备对象，表示基于
    系统能力选择的CPU、CUDA或MPS。

    Returns:
        torch.device: 表示所选张量计算设备的设备对象。
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return torch.device(device)


def get_logger(log_file: Path, with_time: bool = True):
    """配置一个记录器，将消息输出到指定文件和标准错误。

    该函数初始化一个记录器，可以以DEBUG级别记录消息，将输出
    定向到指定的日志文件和标准错误。它允许在日志消息中可选地
    包含时间戳，从而增强日志的可读性和可追溯性。

    Args:
        log_file (Path): 用于记录日志消息的文件路径。
        with_time (bool): 一个标志，指示是否在日志格式中包含时间戳
        （默认值：True）。

    Returns:
        logger: 配置好的记录器实例，用于记录消息。
    """
    global logger

    logger.remove()
    logger.add(
        log_file,
        level="DEBUG",
        format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ {{message}}",
    )
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ <level>{{message}}</level>",
    )

    return logger
