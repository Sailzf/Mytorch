import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from .distributions import Distribution, UniformDistribution, LogUniformDistribution
from .utils import gaussian_kernel, estimate_density

class TPE:
    def __init__(
        self,
        search_space: Dict[str, Distribution],
        n_ei_candidates: int = 24,
        gamma: float = 0.25,
        seed: Optional[int] = None
    ):
        """Tree-structured Parzen Estimators (TPE) 优化器

        Args:
            search_space: 参数搜索空间，键为参数名，值为分布
            n_ei_candidates: 每次采样的候选点数量
            gamma: 将观测值分为好坏两组的比例
            seed: 随机种子
        """
        self.search_space = search_space
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        
        self.trials: List[Dict] = []
        self.values: List[float] = []
        
    def suggest(self) -> Dict[str, Union[float, int]]:
        """生成下一组参数建议"""
        if len(self.trials) < self.n_ei_candidates:
            # 如果样本不足，使用随机采样
            return self._random_sample()
            
        # 将观测值按照性能排序并分组
        sorted_indices = np.argsort(self.values)
        n_below = max(int(len(self.trials) * self.gamma), 1)
        
        below_params = [self.trials[i] for i in sorted_indices[:n_below]]
        above_params = [self.trials[i] for i in sorted_indices[n_below:]]
        
        # 为每个参数估计概率密度
        best_score = float('-inf')
        best_params = None
        
        candidates = [self._random_sample() for _ in range(self.n_ei_candidates)]
        
        for candidate in candidates:
            score = 1.0
            for param_name, param_value in candidate.items():
                below_values = [p[param_name] for p in below_params]
                above_values = [p[param_name] for p in above_params]
                
                l_x = estimate_density(below_values, param_value)
                g_x = estimate_density(above_values, param_value)
                
                # 计算EI
                score *= (l_x + 1e-12) / (g_x + 1e-12)
            
            if score > best_score:
                best_score = score
                best_params = candidate
                
        return best_params
    
    def observe(self, params: Dict[str, Union[float, int]], value: float):
        """观测一组参数的评估结果

        Args:
            params: 参数字典
            value: 评估分数
        """
        self.trials.append(params)
        self.values.append(value)
    
    def _random_sample(self) -> Dict[str, Union[float, int]]:
        """从搜索空间中随机采样一组参数"""
        return {
            name: dist.sample(self.rng)
            for name, dist in self.search_space.items()
        } 