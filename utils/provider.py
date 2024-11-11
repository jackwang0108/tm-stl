"""
点云处理相关函数

    @Time    : 2024/11/8
    @Author  : JackWang
    @File    : provider.py
    @IDE     : VsCode
"""

# Third-Party Library
import numpy as np


def normalize_data(batch_data: np.ndarray) -> np.ndarray:
    """
    对批量数据进行归一化处理，使其中心位于原点。

    该函数接受一个形状为 BxNxC 的数组，计算每个批次的质心，并将点云数据中心化。然后，它将数据缩放，使得每个批次的点云数据的最大距离为1，返回归一化后的数据。

    Args:
        batch_data (np.ndarray): 一个形状为 BxNxC 的 NumPy 数组，其中 B 是批次大小，N 是每个批次中的点数，C 是每个点的特征维度。

    Returns:
        np.ndarray: 归一化后的数据，形状仍为 BxNxC。
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    随机打乱输入数据及其对应的标签。

    该函数随机打乱数据和标签，同时保持它们之间的对应关系。它返回打乱后的数据、打乱后的标签以及用于打乱的索引。

    Args:
        data (np.ndarray): 一个形状为 (B, N, ...) 的 NumPy 数组，表示输入数据。
        labels (np.ndarray): 一个形状为 (B, ...) 的 NumPy 数组，表示数据对应的标签。

    Returns:
        tuple[np.ndarray, np.ndarray]: 一个元组，包含打乱后的数据和打乱后的标签，以及用于打乱的索引。
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data: np.ndarray) -> np.ndarray:
    """
    随机打乱每个点云中点的顺序。

    该函数在每个点云中随机打乱点的顺序，同时对整个批次使用相同的打乱索引。这确保了批次中所有点云的点相对顺序一致地改变。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, C) 的 NumPy 数组，其中 B 是批次大小，N 是点的数量，C 是每个点的特征维度。

    Returns:
        np.ndarray: 一个形状与输入相同的 NumPy 数组，点在每个点云中被打乱。
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotate_point_cloud(batch_data: np.ndarray) -> np.ndarray:
    """
    随机旋转点云以增强数据集。

    该函数对每个点云进行随机旋转，旋转是基于形状的，并沿着上方向进行。返回旋转后的点云批次。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 3) 的 NumPy 数组，表示原始点云的批次，其中 B 是批次大小，N 是每个点云中的点数。

    Returns:
        np.ndarray: 一个形状为 (B, N, 3) 的 NumPy 数组，表示旋转后的点云批次。
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cos = np.cos(rotation_angle)
        sin = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data: np.ndarray) -> np.ndarray:
    """
    随机围绕 Z 轴旋转点云以增强数据集。

    该函数对批次中的每个点云进行围绕 Z 轴的随机旋转。旋转是通过旋转矩阵实现的，这有助于通过提供原始点云的变体来增强数据集。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 3) 的 NumPy 数组，表示原始点云的批次，其中 B 是批次大小，N 是每个点云中的点数。

    Returns:
        np.ndarray: 一个形状为 (B, N, 3) 的 NumPy 数组，表示旋转后的点云批次。
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cos = np.cos(rotation_angle)
        sin = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal: np.ndarray) -> np.ndarray:
    """
    随机旋转包含 XYZ 坐标和法向量的点云。

    该函数对批次中的每个点云进行随机旋转，同时旋转点的法线。输入的点云数据包含 XYZ 坐标和法向量信息，输出为旋转后的点云数据。

    Args:
        batch_xyz_normal (np.ndarray): 一个形状为 (B, N, 6) 的 NumPy 数组，其中 B 是批次大小，N 是每个点云中的点数，前 3 个通道为 XYZ 坐标，后 3 个通道为法向量。

    Returns:
        np.ndarray: 一个形状为 (B, N, 6) 的 NumPy 数组，表示旋转后的 XYZ 坐标和法向量的点云。
    """
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cos = np.cos(rotation_angle)
        sin = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(
            shape_normal.reshape((-1, 3)), rotation_matrix
        )
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(
    batch_data: np.ndarray, angle_sigma: float = 0.06, angle_clip: float = 0.18
) -> np.ndarray:
    """
    随机扰动点云，通过小旋转增强数据集。

    该函数对批次中的每个点云进行小幅随机旋转，同时旋转点的法线。通过添加扰动，增强数据集的多样性，帮助提高模型的鲁棒性。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 6) 的 NumPy 数组，其中 B 是批次大小，N 是每个点云中的点数，前 3 个通道为 XYZ 坐标，后 3 个通道为法线。
        angle_sigma (float): 控制旋转扰动的标准差，默认值为 0.06。
        angle_clip (float): 限制旋转角度的最大值，默认值为 0.18。

    Returns:
        np.ndarray: 一个形状为 (B, N, 6) 的 NumPy 数组，表示经过扰动后的点云和法向量。
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(
    batch_data: np.ndarray, rotation_angle: float
) -> np.ndarray:
    """
    按照指定角度旋转点云。

    该函数对批次中的每个点云进行围绕上方向的旋转，旋转角度由参数指定。返回旋转后的点云数据，增强数据集的多样性。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 3) 的 NumPy 数组，表示原始点云的批次，其中 B 是批次大小，N 是每个点云中的点数。
        rotation_angle (float): 旋转的角度，以弧度为单位。

    Returns:
        np.ndarray: 一个形状为 (B, N, 3) 的 NumPy 数组，表示旋转后的点云批次。
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cos = np.cos(rotation_angle)
        sin = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(
    batch_data: np.ndarray, rotation_angle: float
) -> np.ndarray:
    """
    按照指定角度旋转点云及其法向量。

    该函数对批次中的每个点云及其对应的法向量进行围绕上方向的旋转，旋转角度由参数指定。旋转是围绕 Y 轴进行的，确保点的位置和法向量都得到一致的调整。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 6) 的 NumPy 数组，其中 B 是批次大小，N 是每个点云中的点数，前 3 个通道表示 XYZ 坐标，后 3 个通道表示法线。
        rotation_angle (float): 旋转的角度，以弧度为单位。

    Returns:
        np.ndarray: 一个形状为 (B, N, 6) 的 NumPy 数组，表示旋转后的点云及法向量。
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cos = np.cos(rotation_angle)
        sin = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(
    batch_data: np.ndarray, angle_sigma: float = 0.06, angle_clip: float = 0.18
) -> np.ndarray:
    """
    随机扰动点云，通过小旋转增强数据集。

    该函数对批次中的每个点云进行小幅随机旋转，旋转角度由正态分布生成，并受到限制。通过添加扰动，增强数据集的多样性，帮助提高模型的鲁棒性。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 3) 的 NumPy 数组，表示原始点云的批次，其中 B 是批次大小，N 是每个点云中的点数。
        angle_sigma (float): 控制旋转扰动的标准差，默认值为 0.06。
        angle_clip (float): 限制旋转角度的最大值，默认值为 0.18。

    Returns:
        np.ndarray: 一个形状为 (B, N, 3) 的 NumPy 数组，表示经过扰动后的点云。
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data: np.ndarray, sigma=0.01, clip=0.05) -> np.ndarray:
    """
    随机抖动点云中的每个点，以增强数据集。

    该函数对批次中的每个点进行小幅随机抖动，抖动的幅度由参数 sigma 控制，并且可以通过 clip 限制抖动的最大值。通过添加抖动，增强数据集的多样性，帮助提高模型的鲁棒性。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 3) 的 NumPy 数组，表示原始点云的批次，其中 B 是批次大小，N 是每个点云中的点数。
        sigma (float): 控制抖动幅度的标准差，默认值为 0.01。
        clip (float): 限制抖动幅度的最大值，默认值为 0.05。

    Returns:
        np.ndarray: 一个形状为 (B, N, 3) 的 NumPy 数组，表示经过抖动处理后的点云。
    """
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
    """
    随机平移点云，以增强数据集。

    该函数对批次中的每个点云进行随机平移，平移的范围由参数 shift_range 控制。通过添加平移，增强数据集的多样性，帮助提高模型的鲁棒性。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 3) 的 NumPy 数组，表示原始点云的批次，其中 B 是批次大小，N 是每个点云中的点数。
        shift_range (float): 控制平移范围的最大值，默认值为 0.1。

    Returns:
        np.ndarray: 一个形状为 (B, N, 3) 的 NumPy 数组，表示经过平移处理后的点云。
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(
    batch_data: np.ndarray, scale_low: float = 0.8, scale_high: float = 1.25
) -> np.ndarray:
    """
    随机缩放点云，以增强数据集。

    该函数对批次中的每个点云进行随机缩放，缩放因子在指定的范围内生成。通过添加缩放变换，增强数据集的多样性，帮助提高模型的鲁棒性。

    Args:
        batch_data (np.ndarray): 一个形状为 (B, N, 3) 的 NumPy 数组，表示原始点云的批次，其中 B 是批次大小，N 是每个点云中的点数。
        scale_low (float): 缩放因子的下限，默认值为 0.8。
        scale_high (float): 缩放因子的上限，默认值为 1.25。

    Returns:
        np.ndarray: 一个形状为 (B, N, 3) 的 NumPy 数组，表示经过缩放处理后的点云。
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(
    batch_pc: np.ndarray, max_dropout_ratio: float = 0.875
) -> np.ndarray:
    """
    随机丢弃点云中的部分点，以增强数据集的鲁棒性。

    该函数对批次中的每个点云随机丢弃一定比例的点，丢弃的比例由参数 max_dropout_ratio 控制。被丢弃的点将被第一个点的坐标替代，从而保持点云的结构。

    Args:
        batch_pc (np.ndarray): 一个形状为 (B, N, 3) 的 NumPy 数组，表示原始点云的批次，其中 B 是批次大小，N 是每个点云中的点数。
        max_dropout_ratio (float): 最大丢弃比例，默认值为 0.875。

    Returns:
        np.ndarray: 一个形状为 (B, N, 3) 的 NumPy 数组，表示经过丢弃处理后的点云。
    """
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc
