# Standard Library
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# Third-Party Library
import tqdm
import pandas as pd


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="STL数据集整理工具")
    parser.add_argument(
        "-i",
        "--ipath",
        type=str,
        required=True,
        help="源STL数据文件夹",
    )

    parser.add_argument(
        "-o",
        "--opath",
        type=str,
        default="./output",
        help="目标STL数据文件夹",
    )

    return parser.parse_args()


def get_filelist(root_dir: Path) -> list[Path]:

    if not root_dir.is_dir():
        return [root_dir]

    filelist = []
    for root_dir in root_dir.iterdir():
        filelist.extend(get_filelist(root_dir))

    return filelist


def sort_filelist(filelist: list[Path]) -> dict[str, list[Path]]:

    sorted_filelist: dict[str, list[Path]] = defaultdict(list)

    for file in filelist:
        sorted_filelist[file.parts[1]].append(file)

    return sorted_filelist


def get_file_mapping(original_dir: Path, target_dir: Path) -> pd.DataFrame:
    assert original_dir.is_dir()

    files = sort_filelist(get_filelist(original_dir))

    mappings = []
    for type, filenames in files.items():
        mappings.extend(
            {"original": file, "new": target_dir / f"{type}/{idx}.stl"}
            for idx, file in enumerate(filenames)
        )
    return pd.DataFrame.from_dict(mappings)


def copy_files(file_mappings: pd.DataFrame):
    target: Path
    original: Path
    for idx, (original, target) in (
        tbar := tqdm.tqdm(file_mappings.iterrows(), total=file_mappings.shape[0])
    ):
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(original, target)
        tbar.write(f"Copy {original} ---> {target}")


def main():
    args = get_args()
    source_dir, target_dir = Path(args.ipath), Path(args.opath)

    assert source_dir.exists(), f"{source_dir} 不存在"

    file_mappings = get_file_mapping(source_dir, target_dir)
    copy_files(file_mappings)
    file_mappings.to_csv(f := target_dir / "文件映射.csv")
    print(f"文件映射已保存至 {f}")


if __name__ == "__main__":
    main()
