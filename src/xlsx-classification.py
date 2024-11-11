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
    calculate_acc,
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
xlsx_path = base_dir / "data/数据集10.25强度-分类.xlsx"
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

    batch_x: torch.FloatTensor
    batch_y: torch.FloatTensor
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = (
            batch_x.to(device, torch.float),
            batch_y.to(device, torch.int).squeeze(),
        )

        optimizer.zero_grad()

        output = model(batch_x)
        loss: torch.FloatTensor = loss_func(output, batch_y)

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

        x, y = x.to(device, torch.float), y.to(device, torch.int).squeeze()
        pred = model(x)
        performances.append(calculate_acc(y, pred))

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

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    best_model = None
    best_val_perf = 0
    for epoch in range(num_epoch):

        train_loss = train_epoch(model, train_loader, loss_func, optimizer)

        val_perf = val_epoch(model, val_loader, calculate_acc)

        if best_val_perf < val_perf:
            best_val_perf = val_perf
            best_model = copy.deepcopy(model)

        digits = len(f"{num_epoch}")
        logger.info(
            f"Epoch: [{epoch:{digits}d}/{num_epoch}], train loss: {train_loss:.4f}, validation Acc: {val_perf*100:.2f} %, best validation Acc: {best_val_perf*100:.2f} %"
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

        x, y = x.to(device, torch.float), y.to(device, torch.int).squeeze()
        pred = model(x)
        performaces.append(calculate_acc(y, pred))

    r2 = sum(performaces) / len(performaces)
    logger.info(f"Testing R^2: {r2:.4f}")


def exp1():
    """
    把强度作为特征, 单纯用于分类
    """

    # 最后一行是类别, 拆分出来的x是将强度作为特征的
    x, y = read_excel(xlsx_path, target_col=-1)
    train_data, val_data, test_data = random_split_data(x, y, ratio=[7, 1, 2])

    # Linear, 78.12% 的准确率
    # model = LinearNetwork(
    #     in_features=x.shape[1], out_features=4, hidden_dim=8, num_layers=1
    # ).to(device)

    # Conv1D, 81.25% 的准确率
    # model = Conv1DRegression(input_size=x.shape[1], output_size=4).to(device)

    # Residual 96.88% 的准确率
    # model = LinearResidualNetwork(
    #     in_features=x.shape[1], out_features=4, hidden_dim=16, num_layers=3
    # ).to(device)

    # ResidualAttention 93.75% 的准确率
    model = LinearResidualAttentionNetwork(
        in_features=x.shape[1], out_features=4, hidden_dim=16, num_layers=1
    ).to(device)

    model = train(model, train_data, val_data, 400)


def exp2():
    """
    只使用几何特征, 不用强度特征, 单纯用于分类
    """

    # 最后一行是类别, 拆分出来的x是将强度作为特征的
    x, y = read_excel(xlsx_path, target_col=-1)
    x = x[:, :-1]
    train_data, val_data, test_data = random_split_data(x, y, ratio=[7, 1, 2])

    # Linear, 81,25% 的准确率
    # model = LinearNetwork(
    #     in_features=x.shape[1], out_features=4, hidden_dim=8, num_layers=1
    # ).to(device)

    # Conv1D, 81.25% 的准确率
    # model = Conv1DRegression(input_size=x.shape[1], output_size=4).to(device)

    # Residual 96.88% 的准确率
    # model = LinearResidualNetwork(
    #     in_features=x.shape[1], out_features=4, hidden_dim=16, num_layers=3
    # ).to(device)

    # ResidualAttention 96.88% 的准确率
    model = LinearResidualAttentionNetwork(
        in_features=x.shape[1], out_features=4, hidden_dim=16, num_layers=1
    ).to(device)

    model = train(model, train_data, val_data, 400)


if __name__ == "__main__":
    exp2()
