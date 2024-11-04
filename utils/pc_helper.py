# Standard Library
from pathlib import Path

# Third-Party Library
import numpy as np
import open3d as o3d

# Torch Library

# My Library


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


if __name__ == "__main__":
    a = read_pointcloud(
        Path(__file__).parent.resolve() / "../data/old-sorted-pc/Brock/0.pcd"
    )
    print(a.shape)
