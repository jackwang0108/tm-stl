# Standard Library
from pathlib import Path
from functools import lru_cache

# Third-Party Library
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Torch Library
import torch
from torch.utils.data import Dataset

# My Library


def calculate_r2(gt: torch.FloatTensor, pred: torch.FloatTensor) -> float:
    """计算R平方（决定系数）值。

    该函数计算R平方值，这是一个统计度量，表示回归模型中自变量或变量解释的因变量方差的比例。它接受真实值和预测值作为输入，并返回R平方值作为浮点数。

    Args:
        gt (torch.FloatTensor): 真实值。
        pred (torch.FloatTensor): 预测值。

    Returns:
        float: 表示模型拟合优度的R平方值。

    Raises:
        ValueError: 如果输入张量的形状不相同。
    """
    ss_tot = ((gt - gt.mean(dim=0)) ** 2).sum()
    ss_res = ((gt - pred) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def calculate_acc(gt: torch.FloatTensor, pred: torch.FloatTensor) -> float:
    return ((pred.argmax(dim=1) == gt.squeeze()).sum() / gt.shape[0]).item()


class XlsxDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__()

        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.x[index], self.y[index]


def collate_fn(
    batched_data: list[tuple[np.ndarray, np.float32]]
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """将一批数据整理为PyTorch张量。

    该函数接收一个包含特征数组和对应目标值的元组列表，并将它们
    转换为PyTorch张量。特征被堆叠成一个单一的张量，而目标值也
    被堆叠并重塑，以确保它们具有正确的维度。

    Args:
        batched_data (list[tuple[np.ndarray, np.float32]]): 一个元组列表，
        每个元组包含一个特征数组和一个目标值。

    Returns:
        tuple[torch.FloatTensor, torch.FloatTensor]: 包含两个PyTorch张量的元组：
        第一个张量表示堆叠后的特征，第二个张量表示堆叠后的目标值。
    """
    batch_x = torch.from_numpy(np.stack([batch[0] for batch in batched_data], axis=0))
    batch_y = torch.from_numpy(
        np.stack([batch[1] for batch in batched_data], axis=0)
    ).unsqueeze(dim=-1)
    return batch_x, batch_y


@lru_cache(maxsize=3)
def read_excel(path: Path, target_col: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """读取Excel文件并提取特征和目标变量。

    该函数从Excel文件中加载数据，并将指定的目标列与其余数据分开。它返回特征作为NumPy数组，目标变量作为另一个NumPy数组。

    Args:
        path (Path): 要读取的Excel文件的路径。
        target_col (int): 要提取的目标列的索引（默认值：-1，表示最后一列）。

    Returns:
        tuple[np.ndarray, np.ndarray]: 包含两个NumPy数组的元组：第一个数组表示特征，第二个数组表示目标变量。

    Raises:
        AssertionError: 如果指定的文件不存在或目标列索引超出范围。
    """
    assert path.exists(), f"文件不存在, {path=}"

    df = pd.read_excel(path, header=0)
    assert (
        target_col < df.shape[-1]
    ), f"指定的列 {target_col=} 不存在, xlsx表格大小: {df.shape=}"

    y = df.iloc[:, target_col]
    x = df.drop(df.columns[target_col], axis=1)
    return x.to_numpy(), y.to_numpy()


def random_split_data(x: np.ndarray, y: np.ndarray, ratio: list[int | float] = None):
    """随机将数据拆分为训练集、验证集和测试集。

    该函数接收特征数组和目标数组，并根据指定的比例将它们拆分为
    训练集、验证集和测试集。如果未提供比例，则默认为70-10-20的拆分。

    Args:
        x (np.ndarray): 要拆分的特征数据。
        y (np.ndarray): 与特征对应的目标数据。
        ratio (list[int | float], optional): 指定训练集、验证集和测试集
        拆分比例的列表。默认为 [7, 1, 2]。

    Returns:
        tuple: 包含三个数组对的元组：
        （训练特征，训练目标），
        （验证特征，验证目标），
        （测试特征，测试目标）。
    """

    if ratio is None:
        ratio = [7, 1, 2]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=ratio[-1] / sum(ratio)
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=ratio[-2] / sum(ratio[:-1]), random_state=42
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def k_fold_split_index(
    x: np.ndarray, y: np.ndarray, k: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """将数据集拆分为 K 套训练集、验证集和测试集

    参数:
        x: 特征数据，numpy 数组或类似结构
        y: 标签数据，numpy 数组或类似结构
        k: 交叉验证的折数

    返回:
        splits: 一个包含 K 套 (train_indices, val_indices, test_indices) 的列表
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    splits = []
    for train_val_indices, test_indices in kf.split(x):
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.2, random_state=42
        )
        splits.append((train_indices, val_indices, test_indices))

    return splits


if __name__ == "__main__":
    xlsx_path = Path(__file__).resolve().parent / "../data/数据集10.25强度.xlsx"
    x, y = read_excel(xlsx_path, -1)

    from torch.utils.data import DataLoader

    dataset = XlsxDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    for batch_x, batch_y in loader:
        print(x.shape)
