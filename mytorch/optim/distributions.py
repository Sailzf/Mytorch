from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class Distribution(ABC):
    """参数分布的抽象基类"""
    
    @abstractmethod
    def sample(self, rng: np.random.RandomState):
        """从分布中采样一个值"""
        pass

class UniformDistribution(Distribution):
    """均匀分布"""
    
    def __init__(self, low: float, high: float):
        """
        Args:
            low: 下界
            high: 上界
        """
        self.low = low
        self.high = high
        
    def sample(self, rng: np.random.RandomState):
        return float(rng.uniform(self.low, self.high))

class LogUniformDistribution(Distribution):
    """对数均匀分布"""
    
    def __init__(self, low: float, high: float):
        """
        Args:
            low: 下界
            high: 上界
        """
        self.low = np.log(low)
        self.high = np.log(high)
        
    def sample(self, rng: np.random.RandomState):
        return float(np.exp(rng.uniform(self.low, self.high)))

class IntUniformDistribution(Distribution):
    """整数均匀分布"""
    
    def __init__(self, low: int, high: int):
        """
        Args:
            low: 下界（包含）
            high: 上界（包含）
        """
        self.low = low
        self.high = high
        
    def sample(self, rng: np.random.RandomState):
        return int(rng.randint(self.low, self.high + 1))

class CategoricalDistribution(Distribution):
    """类别分布"""
    
    def __init__(self, choices):
        """
        Args:
            choices: 可选值列表
        """
        self.choices = choices
        
    def sample(self, rng: np.random.RandomState):
        return rng.choice(self.choices) 