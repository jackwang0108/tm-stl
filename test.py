import numpy as np
from sklearn.model_selection import KFold


def k_fold_split(x_all, y_all, k):
    """
    将数据集拆分为 K 套训练集、验证集和测试集

    参数:
    x_all: 特征数据，numpy 数组或类似结构
    y_all: 标签数据，numpy 数组或类似结构
    k: 交叉验证的折数

    返回:
    splits: 一个包含 K 套 (train_indices, val_indices, test_indices) 的列表
    """
    # 确保 x_all 和 y_all 是 numpy 数组
    x_all = np.array(x_all)
    y_all = np.array(y_all)

    # 创建 KFold 对象
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    splits = []

    # 进行 K-fold 拆分
    for train_val_indices, test_indices in kf.split(x_all):
        # 在训练集和验证集之间进行拆分
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.2, random_state=42
        )

        # 保存当前的训练、验证和测试集索引
        splits.append((train_indices, val_indices, test_indices))

    return splits


def train_test_split(indices, test_size=0.2, random_state=None):
    """
    自定义的 train_test_split 函数，用于从给定索引中拆分出训练集和验证集
    """
    np.random.seed(random_state)
    np.random.shuffle(indices)

    split_point = int(len(indices) * (1 - test_size))

    return indices[:split_point], indices[split_point:]


# 示例用法
x_all = np.random.rand(100, 10)  # 100 个样本，10 个特征
y_all = np.random.randint(0, 2, size=100)  # 100 个样本的二分类标签
k = 5  # 设置 K

splits = k_fold_split(x_all, y_all, k)

# 输出每个拆分的索引
for i, (train_indices, val_indices, test_indices) in enumerate(splits):
    print(f"Fold {i + 1}:")
    print(f"  Train indices: {train_indices}")
    print(f"  Validation indices: {val_indices}")
    print(f"  Test indices: {test_indices}")
