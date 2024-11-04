# Standard Library
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count


# Third-Party Library
import tqdm

import open3d as o3d


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="STL数据转点云工具")
    parser.add_argument(
        "-i",
        "--ipath",
        type=str,
        required=True,
        help="STL文件或文件夹路径",
    )
    parser.add_argument(
        "-o",
        "--opath",
        type=str,
        default="./point/cloud",
        help="STL文件或文件夹路径",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=30000,
        help="生成的点云中点的数量",
    )

    parser.add_argument(
        "-c",
        "--cpu",
        type=int,
        default=cpu_count() // 4,
        help="并行调用的CPU数量. PS: 太大会导致系统卡死",
    )

    return parser.parse_args()


def get_filelist(root: Path) -> list[Path]:

    if not root.is_dir():
        return [root]

    filelist = []
    for subdir in root.iterdir():
        filelist.extend(get_filelist(subdir))

    return filelist


def get_tasks(
    source_dir: Path, target_dir: Path, filelist: list[Path], num_points: int
) -> tuple[Path, Path, int]:

    return [
        (
            stl_path,
            target_dir / stl_path.relative_to(source_dir).with_suffix(".pcd"),
            num_points,
        )
        for stl_path in filelist
        if stl_path.suffix == ".stl"
    ]


def stl2pc(args: tuple[Path, Path, int]):
    stl_path, pc_path, num_points = args

    assert stl_path.exists(), f"{stl_path}不存在"

    try:
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        mesh.compute_vertex_normals()

        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

        pc_path.parent.mkdir(exist_ok=True, parents=True)
        o3d.io.write_point_cloud(str(pc_path), pcd)
    except Exception as e:
        print(e)
        print(f"{stl_path=}, {pc_path=}, {num_points=}")

    return stl_path, pc_path, num_points


def main():

    args = get_args()

    source_dir = Path(args.ipath).resolve()
    target_dir = Path(args.opath).resolve()

    stl_filelist = get_filelist(source_dir)
    tasks = get_tasks(source_dir, target_dir, stl_filelist, args.num)

    with Pool(processes=args.cpu) as pool, tqdm.tqdm(total=len(tasks)) as tbar:
        for stl_path, pc_path, num in pool.imap(stl2pc, tasks):
            tqdm.tqdm.write(
                f"Convert {stl_path.relative_to(source_dir.parent)} to {pc_path.relative_to(target_dir.parent)} with {num} points"
            )
            tbar.update(1)


if __name__ == "__main__":
    main()
