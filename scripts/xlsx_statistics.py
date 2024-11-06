# Standard Library
from pathlib import Path
from typing import Literal
from functools import lru_cache
from argparse import ArgumentParser, Namespace

# Third-Party Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# My Library

xlsx_path: Path = None
output_path: Path = None


def get_args() -> Namespace:
    parser = ArgumentParser("对XLSX表格数据进行数据分析")
    parser.add_argument(
        "-i",
        "--ipath",
        type=str,
        required=True,
        help="XLSX表格文件路径",
    )
    parser.add_argument(
        "-o",
        "--opath",
        type=str,
        default=Path(__file__).resolve().parent.parent,
        help="分析结果输出路径",
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["linkage", "pca", "tsne"],
        default="linkage",
        help="进行的数据分析类型",
    )

    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=2,
        help="降维可视化的维度数量 (2维或3维), 3维默认无法保存图像",
    )

    return parser.parse_args()


@lru_cache(maxsize=3)
def read_excel(path: Path) -> np.ndarray:
    assert path.exists(), f"文件不存在, {path=}"
    return pd.read_excel(path, header=0)


def calculate_statistics(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return data.mean(axis=0), data.std(axis=0)


class XlsxTool:

    def __init__(self, xlsx_path: Path):
        self.xlsx_path: Path = xlsx_path

        self.data: pd.DataFrame = read_excel(xlsx_path)
        self.feature_names: pd.Series = self.data.columns

        self.mean, self.std = calculate_statistics(self.data)

    def _check_type(self, x: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
        x = x.to_numpy() if isinstance(x, (pd.DataFrame, pd.Series)) else x
        x = x if isinstance(x, np.ndarray) else x.cpu().numpy()
        x = np.expand_dims(x, 0) if x.ndim == 1 else x
        return x

    def norm(
        self,
        x: np.ndarray | pd.DataFrame | pd.Series,
        scope: Literal["feature", "label", "all"] = "all",
    ) -> np.ndarray:
        # x = self._check_type(x)
        if scope == "feature":
            start, end = 0, -1
        elif scope == "label":
            start, end = -1, -1
        else:
            start, end = 0, len(self.mean)
        x = (x - self.mean.to_numpy()[start:end]) / (
            self.std.to_numpy()[start:end] + 1e-9
        )
        return x

    def denorm(
        self,
        x: np.ndarray | pd.DataFrame | pd.Series,
        scope: Literal["feature", "label", "all"] = "all",
    ) -> np.ndarray:
        x = self._check_type(x)
        if scope == "feature":
            start, end = 0, -1
        elif scope == "label":
            start, end = -1, -1
        else:
            start, end = 0, len(self.mean)
        x = x * self.std.to_numpy()[start:end] + self.mean.to_numpy()[start:end]
        return x

    def plot_linkage(
        self,
        image_path: Path = None,
        row_cluster: bool = False,
        col_cluster: bool = True,
    ):
        """进行层次聚类分析并绘制树状图"""
        normed_data = self.norm(self.data.copy().to_numpy()[:, :-1], "feature")
        sns.set_theme(style="white")
        sns.clustermap(
            normed_data,
            method="ward",
            cmap="viridis",
            figsize=(15, 10),
            standard_scale=1,
            col_cluster=col_cluster,
            row_cluster=row_cluster,
        ).figure.suptitle("Hierarchical Clustering Heatmap", y=0.999)

        plt.savefig(
            image_path if image_path is not None else f"./{self.plot_linkage.__name__}",
            dpi=300,
        )

    def plot2d(self, data: pd.DataFrame, **kwargs):
        plt.clf()
        plt.figure(figsize=(16, 10))
        sns.scatterplot(data=data, **kwargs)

    def plot_pca(
        self,
        image_path: Path = None,
        n_components: int = 3,
        target_col: int = -1,
        scope: Literal["feature", "all"] = "all",
    ) -> pd.DataFrame:
        # sourcery skip: assign-if-exp, extract-method, switch
        """使用PCA主成分分析进行降维"""

        normed_data = self.norm(self.data.copy().to_numpy()[:, :-1], "feature")
        if scope == "feature":
            start, end = 0, -1
        else:
            start, end = 0, len(self.mean)

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(normed_data[:, start:end])
        pca_result = pd.DataFrame.from_dict(
            {
                f"PCA Component {i+1}": pca_result[:, i]
                for i in range(pca_result.shape[-1])
            }
        )
        t = self.data.to_numpy()[:, target_col]
        pca_result["intensity-size"] = (t - t.min()) / (t.max() - t.min()) * 150
        pca_result["intensity-color"] = (t - t.min()) / (t.max() - t.min())

        plt.clf()
        if n_components == 2:
            self.plot2d(
                data=pca_result,
                x="PCA Component 1",
                y="PCA Component 2",
                hue="intensity-color",
                palette="viridis",
                legend="auto",
                alpha=0.7,
                size="intensity-size",
                sizes=(20, 200),
            )

            plt.savefig(
                image_path if image_path is not None else f"./{self.plot_pca.__name__}",
                dpi=300,
            )
        elif n_components == 3:
            ax = plt.figure(figsize=(16, 10)).add_subplot(projection="3d")
            cmap = plt.get_cmap("viridis")
            norm = Normalize(
                vmin=pca_result["intensity-color"].min(),
                vmax=pca_result["intensity-color"].max(),
            )
            ax.scatter(
                xs=pca_result.loc[:, "PCA Component 1"],
                ys=pca_result.loc[:, "PCA Component 2"],
                zs=pca_result.loc[:, "PCA Component 3"],
                s=pca_result["intensity-size"],
                c=pca_result["intensity-color"],
                cmap=cmap,
                norm=norm,
            )
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_zlabel("PCA Component 3")
            cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.1)
            cbar.set_label("Intensity Color")
            plt.show()

        return pca_result

    def plot_tsne(
        self,
        image_path: Path = None,
        n_components: int = 3,
        target_col: int = -1,
        scope: Literal["feature", "all"] = "all",
    ) -> pd.DataFrame:
        # sourcery skip: assign-if-exp, extract-method, switch
        normed_data = self.norm(self.data.copy().to_numpy()[:, :-1], "feature")
        if scope == "feature":
            start, end = 0, -1
        else:
            start, end = 0, len(self.mean)

        tsne = TSNE(n_components=n_components)
        tsne_result = tsne.fit_transform(normed_data[:, start:end])
        tsne_result = pd.DataFrame.from_dict(
            {
                f"PCA Component {i+1}": tsne_result[:, i]
                for i in range(tsne_result.shape[-1])
            }
        )
        t = self.data.to_numpy()[:, target_col]
        tsne_result["intensity-size"] = (t - t.min()) / (t.max() - t.min()) * 150
        tsne_result["intensity-color"] = (t - t.min()) / (t.max() - t.min())
        # t

        if n_components == 2:
            self.plot2d(
                data=tsne_result,
                x="PCA Component 1",
                y="PCA Component 2",
                hue="intensity-color",
                palette="viridis",
                legend="auto",
                alpha=0.7,
                size="intensity-size",
                sizes=(20, 200),
            )

            plt.savefig(
                (
                    image_path
                    if image_path is not None
                    else f"./{self.plot_tsne.__name__}"
                ),
                dpi=300,
            )
        elif n_components == 3:
            ax = plt.figure(figsize=(16, 10)).add_subplot(projection="3d")
            cmap = plt.get_cmap("viridis")
            norm = Normalize(
                vmin=tsne_result["intensity-color"].min(),
                vmax=tsne_result["intensity-color"].max(),
            )
            ax.scatter(
                xs=tsne_result.loc[:, "PCA Component 1"],
                ys=tsne_result.loc[:, "PCA Component 2"],
                zs=tsne_result.loc[:, "PCA Component 3"],
                s=tsne_result["intensity-size"],
                c=tsne_result["intensity-color"],
                cmap=cmap,
                norm=norm,
            )
            ax.set_xlabel("TSNE Component 1")
            ax.set_ylabel("TSNE Component 2")
            ax.set_zlabel("TSNE Component 3")
            cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.1)
            cbar.set_label("Intensity Color")
            plt.show()

        return tsne_result


if __name__ == "__main__":

    args = get_args()

    xlsx_path = Path(args.ipath)
    output_path = Path(args.opath)

    assert xlsx_path.exists(), f"文件不存在 {xlsx_path}"
    output_path.mkdir(exist_ok=True, parents=True)

    toolbox = XlsxTool(xlsx_path)

    dim = int(args.dim)
    if args.type == "linkage":
        toolbox.plot_linkage(
            row_cluster=True, image_path=output_path / "层次聚类分析图.jpg"
        )
    elif args.type == "pca":
        toolbox.plot_pca(
            n_components=dim, image_path=output_path / "PCA特征降维可视化图.jpg"
        )
    elif args.type == "tsne":
        toolbox.plot_tsne(
            n_components=dim, image_path=output_path / "TSNE特征降维可视化图.jpg"
        )
