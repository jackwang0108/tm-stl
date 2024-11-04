# Standard Library
import copy
import datetime
from pathlib import Path
from collections.abc import Callable

# Third-Party Library
import tqdm
import numpy as np
import pandas as pd

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My Library
from utils import (
    get_device,
    get_logger,
    set_random_seed,
)
from utils.xlsx_helper import (
    read_excel,
    collate_fn,
    calculate_r2,
    random_split_data,
    XlsxDataset,
)
from model.xlsx.linear import LinearNetwork
from model.xlsx.conv1d import Conv1DRegression
from model.xlsx.fcn import FullyConvolutionNetwork
from model.xlsx.linear_residual import LinearResidualNetwork
from model.xlsx.linear_residual_attention import LinearResidualAttentionNetwork


set_random_seed(42)

device = get_device()
base_dir = Path(__file__).resolve().parent.parent
xlsx_path = base_dir / "data/数据集10.25强度.xlsx"
log_dir = base_dir / "log"
logger_path = log_dir / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
logger = get_logger(logger_path)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_func: nn.Module,
    optimizer: optim.Optimizer,
) -> torch.FloatTensor:

    model.train(True)

    x: torch.FloatTensor
    y: torch.FloatTensor
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device, torch.float), y.to(device, torch.float)
        optimizer.zero_grad()

        output = model(x)
        loss: torch.FloatTensor = loss_func(output, y)

        loss.backward()
        optimizer.step()

        digits = len(f"{len(train_loader)}")
        logger.info(f"\t\t batch [{i:{digits}d}/{len(train_loader)}], {loss=:.4f}")

    return loss.clone().detach().cpu()


@torch.no_grad()
def val_epoch(
    model: nn.ModuleList, val_loader: DataLoader, perf_func: Callable
) -> float:

    model.train(False)
    performances = []

    x: torch.FloatTensor
    y: torch.FloatTensor
    for x, y in val_loader:

        x, y = x.to(device, torch.float), y.to(device, torch.float)
        pred = model(x)
        performances.append(calculate_r2(y, pred))

    return sum(performances) / len(performances)


def train(
    model: nn.Module,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    num_epoch: int,
):
    # sourcery skip: min-max-identity

    valset, trainset = XlsxDataset(*val_data), XlsxDataset(*train_data)
    val_loader = DataLoader(valset, 32, shuffle=False, collate_fn=collate_fn)
    train_loader = DataLoader(trainset, 32, shuffle=True, collate_fn=collate_fn)

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    best_model = None
    best_val_perf = 0
    for epoch in range(num_epoch):

        train_loss = train_epoch(model, train_loader, loss_func, optimizer)

        val_perf = val_epoch(model, val_loader, calculate_r2)

        if best_val_perf < val_perf:
            best_val_perf = val_perf
            best_model = copy.deepcopy(model)

        digits = len(f"{num_epoch}")
        logger.info(
            f"Epoch: [{epoch:{digits}d}/{num_epoch}], train loss: {train_loss:.4f}, validation R2: {val_perf:.4f}, best validation R2: {best_val_perf:.4f}"
        )
    return best_model


@torch.no_grad()
def test(model: nn.Module, test_data: tuple[np.ndarray, np.ndarray]):
    model.train(False)

    testset = XlsxDataset(*test_data)
    test_loader = DataLoader(testset, 32, shuffle=False, collate_fn=collate_fn)

    x: torch.FloatTensor
    y: torch.FloatTensor
    performaces = []
    for x, y in test_loader:

        x, y = x.to(device, torch.float), y.to(device, torch.float)
        pred = model(x)
        performaces.append(calculate_r2(y, pred))

    r2 = sum(performaces) / len(performaces)
    logger.info(f"Testing R^2: {r2:.4f}")


def main():

    x, y = read_excel(xlsx_path, target_col=-1)
    train_data, val_data, test_data = random_split_data(x, y, ratio=[7, 1, 2])

    # Linear 主要问题, 大部分参数集中在第一层了, 层数太浅, representation的性能不好
    # model = LinearNetwork(
    #     in_features=24, out_features=1, hidden_dim=8, num_layers=1
    # ).to(device)

    # Conv1D 主要问题, 数据不具有几何结构
    # model = Conv1DRegression(input_size=24, output_size=1).to(device)

    # 全卷积网络
    # model = FullyConvolutionNetwork(24, 1, hidden_dim=8, num_layers=1).to(device)

    # Residual 添加残差来扩展深度, 性能确实提升很多, 但是不乐观
    # model = LinearResidualNetwork(24, 1, hidden_dim=16, num_layers=3).to(device)

    # ResidualAttention 在残差的基础上添加了注意力
    model = LinearResidualAttentionNetwork(24, 1, hidden_dim=16, num_layers=1).to(
        device
    )

    # 大概率是数据本来就有问题

    model = train(model, train_data, val_data, 400)


if __name__ == "__main__":
    main()
