# Standard Library
from pathlib import Path
from collections.abc import Callable
from argparse import ArgumentParser, Namespace

# Third-Party Library
import numpy as np
import open3d as o3d

# Torch Library

# My Library


def get_args() -> Namespace:
    parser = ArgumentParser(usage="可视化STL以及点云数据的工具")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="./data/stl/brock/0.stl",
        help="STL/点云文件路径",
    )
    return parser.parse_args()


def get_abs_path(path: str | Path) -> Path:

    path = Path(path)

    abs_path = Path.cwd() / path
    if len(path.parts) > 0 and path.parts[0] == path.resolve().root:
        abs_path = path

    abs_path = abs_path.resolve()
    assert abs_path.exists(), f"文件不存在 {abs_path}"
    return abs_path


def get_lineset(size: int = 5, step: int = 1):
    def create_grid():
        grid_lines = []
        grid_points = []

        # 生成 XZ 平面上的网格线
        for x in np.arange(-size, size + step, step):
            grid_points.append([x, 0, -size])
            grid_points.append([x, 0, size])
            grid_lines.append([len(grid_points) - 2, len(grid_points) - 1])

        # 生成 YZ 平面上的网格线
        for z in np.arange(-size, size + step, step):
            grid_points.append([0, -size, z])
            grid_points.append([0, size, z])
            grid_lines.append([len(grid_points) - 2, len(grid_points) - 1])

        # 生成 XY 平面上的网格线
        for y in np.arange(-size, size + step, step):
            grid_points.append([-size, y, 0])
            grid_points.append([size, y, 0])
            grid_lines.append([len(grid_points) - 2, len(grid_points) - 1])

        return grid_lines, grid_points

    # 创建网格线
    lines, points = create_grid()

    # 转换为 Open3D 格式
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def main(file: Path):
    assert (s := file.suffix.lower()) in [
        ".stl",
        ".pcd",
    ], f"文件不是STL格式(.stl)/点云格式(.ply), {file}"

    read_func: Callable = (
        o3d.io.read_triangle_mesh if s == ".stl" else o3d.io.read_point_cloud
    )

    data = read_func(str(file))
    if s == ".stl":
        data.compute_vertex_normals()

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    o3d.visualization.draw_geometries([data, coordinate_frame])


if __name__ == "__main__":
    args = get_args()
    main(get_abs_path(args.file))
