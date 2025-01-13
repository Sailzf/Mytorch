import numpy as np
from typing import List, Union

def gaussian_kernel(x: float, points: List[float], bandwidth: float) -> float:
    """计算高斯核密度

    Args:
        x: 需要计算密度的点
        points: 观测点列表
        bandwidth: 带宽参数

    Returns:
        float: 核密度估计值
    """
    points = np.array(points)
    n = len(points)
    if n == 0:
        return 0.0
    
    # Scott's rule for bandwidth selection
    if bandwidth is None:
        bandwidth = 1.059 * np.std(points) * np.power(n, -0.2)
    
    # 避免带宽过小
    bandwidth = max(bandwidth, 1e-6)
    
    # 计算核密度
    diff = (x - points[:, np.newaxis]) / bandwidth
    kernel = np.exp(-0.5 * np.square(diff)) / (np.sqrt(2 * np.pi) * bandwidth)
    return np.mean(kernel)

def estimate_density(points: List[float], x: float, bandwidth: float = None) -> float:
    """使用KDE估计概率密度

    Args:
        points: 观测点列表
        x: 需要估计密度的点
        bandwidth: 带宽参数

    Returns:
        float: 估计的概率密度
    """
    if len(points) == 0:
        return 0.0
    return gaussian_kernel(x, points, bandwidth)

def suggest_bandwidth(points: List[float]) -> float:
    """使用Scott's rule建议带宽参数

    Args:
        points: 观测点列表

    Returns:
        float: 建议的带宽值
    """
    points = np.array(points)
    n = len(points)
    if n < 2:
        return 1.0
    return 1.059 * np.std(points) * np.power(n, -0.2) 