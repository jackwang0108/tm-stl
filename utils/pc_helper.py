# Standard Library
import random
from pathlib import Path
from collections import defaultdict
from typing import Literal, Any, Optional

# Third-Party Library
import numpy as np
import pandas as pd
import open3d as o3d

# Torch Library
import torch
from torch.utils.data import Dataset

# My Library
from .common import make_tupleList


def calculate_acc(gt: torch.FloatTensor, pred: torch.FloatTensor) -> float:
    return ((pred == gt.squeeze()).sum() / gt.shape[0]).item()


def get_pointcloud_files(root: Path) -> dict[str, list[Path]]:
    return {dir.name: list(dir.glob("*.pcd")) for dir in root.iterdir() if dir.is_dir()}


def random_split_data(
    data: list[Any], ratio: list[int | float] = None
) -> tuple[list[Any], list[Any], list[Any]]:

    if ratio is None:
        ratio = [7, 1, 2]

    total_length = len(data)
    split_1 = int(0.7 * total_length)
    split_2 = int(0.9 * total_length)

    random.shuffle(data)

    train, val, test = data[:split_1], data[split_1:split_2], data[split_2:]

    return train, val, test


def read_pointcloud(pc_path: Path, with_color: bool = False) -> np.ndarray:
    """从指定文件路径读取点云数据。

    该函数加载给定文件中的点云，并返回其坐标作为NumPy数组。如果指定，它还可以包含与每个点相关的颜色信息。

    Args:
        pc_path (Path): 要读取的点云文件的路径。
        with_color (bool): 一个标志，指示是否在输出中包含颜色信息（默认值：False）。

    Returns:
        np.ndarray: 形状为[N, 3]的数组，包含点的坐标；如果包含颜色信息，则形状为[N, 6]。

    Raises:
        AssertionError: 如果指定的点云文件不存在。
    """
    assert pc_path.exists(), f"{pc_path} 不存在"
    point_cloud = o3d.io.read_point_cloud(str(pc_path))
    # [N, 3]
    points = np.asarray(point_cloud.points)
    if with_color:
        colors = np.asarray(point_cloud.colors)
        points = (
            points
            if colors.shape[0] == 0
            else np.concatenate((points, colors), axis=-1)
        )
    return points


def normalize_pointcloud(pc: np.ndarray) -> np.ndarray:
    """
    对点云进行归一化

    该函数计算输入点云的质心，将点云中的点围绕该质心中心化，然后进行缩放，使得从质心到任何点的最大距离为1。

    Args:
        pc (np.ndarray): 一个表示点云的 NumPy 数组，其中每一行对应点云中的一个点。

    Returns:
        np.ndarray: 归一化后的点云，作为一个 NumPy 数组，已中心化和缩放。
    """
    centroid: np.ndarray = pc.mean(axis=0)
    centered_pc = pc - centroid
    max_distance = np.sqrt((centered_pc**2).sum(axis=1)).max()
    return centered_pc / max_distance


def farthest_point_sample(pc: np.ndarray, npoint: int) -> np.ndarray:
    """
    通过最远采样, 从点云中采样指定数量的点

    该函数从输入的点云中计算一个点的子集，确保每个选定的点都是距离之前选定点最远的。结果是一个包含采样点的新点云。

    Args:
        point (np.ndarray): 一个表示点云的 NumPy 数组，其中每一行对应一个点。
        npoint (int): 要从点云中采样的点的数量。

    Returns:
        np.ndarray: 一个包含来自原始点云的采样点的 NumPy 数组。
    """
    num_points = pc.shape[0]
    xyz = pc[:, :3]
    selected_points = np.zeros((num_points,))
    distance = np.ones((num_points,)) * np.inf

    # 刚开始随机选择一个点作为降采样开始的中心
    curr_point_idx = np.random.randint(0, num_points)
    for i in range(npoint):
        selected_points[i] = curr_point_idx
        centroid = xyz[curr_point_idx, :]
        dist = ((xyz - centroid) ** 2).sum(axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        curr_point_idx = np.argmax(distance, -1)

    return pc[selected_points.astype(np.int32)]


class PointCloudDataset(Dataset):

    def __init__(
        self,
        files: dict[str, list[Path]],
        num_points: int,
        downsample_policy: Literal["random", "farthest"],
        xlsx_path: Optional[Path] = None,
    ) -> None:
        super().__init__()

        assert isinstance(files, dict), f"错误的数据类型, 预期dict, 实际{type(files)}"
        self.file_list = make_tupleList(files)
        self.cls_mapper = {key: idx for idx, key in enumerate(files.keys())}

        # 采样点的个数
        self.num_points = num_points
        self.downsample_policy = downsample_policy

        # 强度数据
        assert xlsx_path is None or (
            isinstance(xlsx_path, Path) and xlsx_path.exists()
        ), f"{xlsx_path} 不存在"

        self.xlsx_data, self.file_idx_mapper = None, None
        if xlsx_path is not None:
            self.xlsx_data: pd.DataFrame = pd.read_excel(xlsx_path, index_col=0)
            self.file_idx_mapper: dict[str, dict[int, int]] = {}
            self.file_idx_mapper = defaultdict(dict)
            for idx, file in enumerate(self.xlsx_data.loc[:, "目标文件"]):
                file = Path(file)
                self.file_idx_mapper[file.parts[-2]][file.stem] = idx

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:

        pc_path, cls_name = self.file_list[index]

        original_pc = read_pointcloud(pc_path=pc_path)

        # 降采样
        downsample_points = original_pc
        if original_pc.shape[0] > self.num_points:
            if self.downsample_policy == "random":
                downsample_points = original_pc[
                    np.random.choice(
                        original_pc.shape[0], self.num_points, replace=False
                    )
                ]
            elif self.downsample_policy == "farthest":
                downsample_points = farthest_point_sample(original_pc, self.num_points)

        # 归一化
        downsample_points[:, :3] = normalize_pointcloud(downsample_points[:, :3])

        if self.xlsx_data is None:
            return downsample_points, np.array([self.cls_mapper[cls_name]]).astype(
                np.int32
            )

        example_idx = self.file_idx_mapper[pc_path.parts[-2]][pc_path.stem]
        return (
            downsample_points,
            np.array([self.cls_mapper[cls_name]]).astype(np.int32),
            np.array(self.xlsx_data.loc[example_idx, "强度（MPA)"]),
        )


def collate_fn(
    batched_data: list[tuple[np.ndarray, np.ndarray]]
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
    batch_cls = torch.from_numpy(
        np.stack([batch[1] for batch in batched_data], axis=0)
    ).squeeze()

    if len(batched_data[0]) <= 2:
        return batch_x, batch_cls

    batch_reg = torch.from_numpy(
        np.stack([batch[2] for batch in batched_data], axis=0)
    ).squeeze()

    return batch_x, batch_cls, batch_reg


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    root_dir = (Path(__file__).resolve().parent.parent / "data/new-pc").resolve()

    file_dict = get_pointcloud_files(root_dir)
    pc_datasets = PointCloudDataset(
        file_dict,
        num_points=1000,
        downsample_policy="random",
        xlsx_path=Path(__file__).resolve().parent
        / "../data/new-pc/分类-强度数据集.xlsx",
    )

    loader = DataLoader(pc_datasets, batch_size=8, shuffle=True, collate_fn=collate_fn)

    for pc, cls, reg in loader:

        print(pc.shape, cls.shape, reg.shape)
