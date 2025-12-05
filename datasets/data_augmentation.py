import numpy as np
from typing import Callable, Dict, Any, List, Optional


def phase_offset(data:np.ndarray, phase:float)->np.ndarray:
    out = data.copy()
    out[0] = data[0] * np.cos(phase) + data[1] * np.sin(phase)
    out[1] = data[1] * np.cos(phase) - data[0] * np.sin(phase)
    return out


def random_phase_offset(datas:np.ndarray, start:float=-np.pi, end:float=np.pi)->np.ndarray:
    N, C, L = datas.shape
    outs = datas.copy()
    random_numbers = np.random.uniform(start, end, N).reshape(N, 1)
    outs[:, 0] = datas[:, 0] * np.cos(random_numbers) + datas[:, 1] * np.sin(random_numbers)
    outs[:, 1] = datas[:, 1] * np.cos(random_numbers) - datas[:, 0] * np.sin(random_numbers)
    return outs


def stretching(data:np.ndarray, index:int, scale:float)->np.ndarray:
    out = data.copy()
    out[index] = data[index] * scale
    return out


def random_stretching(datas:np.ndarray, start:float=0.75, end:float=1.25)->np.ndarray:
    N, C, L = datas.shape
    outs = datas.copy()
    indexs = np.random.randint(0, C, N)
    scales = np.random.uniform(start, end, N)
    outs[np.arange(N), indexs, :] = datas[np.arange(N), indexs, :] * scales[:, np.newaxis]
    return outs


def mirror_flip(data:np.ndarray, index:int):
    out = data.copy()
    out[index] = data[index] * -1
    return out


def random_mirror_flip(datas:np.ndarray)->np.ndarray:
    N, C, L = datas.shape
    outs = datas.copy()
    indexs = np.random.randint(0, C, N)
    outs[np.arange(N), indexs, :] = datas[np.arange(N), indexs, :] * -1
    return outs


class FeatureAugmentor:
    """
    特征增强器，根据配置自动初始化增强函数。
    参数:
    -----
    config : dict
        每个键为增强方法名，每个值为参数字典。
        例如:
        {
            "random_phase_offset": {"start": -np.pi/2, "end": np.pi/2},
            "random_stretching": {"start": 0.8, "end": 1.2},
            "random_mirror_flip": {}
        }
    global_prob : float
        整体应用增强的概率（0~1）。当随机数大于此值时，整个增强流程会被跳过。
    """
    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None, global_prob: float = 1.0):
        self.config = config or {}
        self.global_prob = global_prob
        self.augmentations: List[Callable[[np.ndarray], np.ndarray]] = []
        self._register_methods()

    def _register_methods(self):
        """根据配置注册增强方法。"""
        method_map = {
            "random_phase_offset": random_phase_offset,
            "random_stretching": random_stretching,
            "random_mirror_flip": random_mirror_flip,
        }
        for name, params in self.config.items():
            if name not in method_map:
                raise ValueError(f"未定义的增强方法: {name}")
            func = method_map[name]
            self.augmentations.append(lambda x, f=func, p=params: f(x, **p))

    def __call__(self, datas: np.ndarray) -> np.ndarray:
        """
        调用对象即执行增强。
        输入:
        -----
        datas : np.ndarray
            输入形状支持:
              - [C, L] 单样本
              - [B, C, L] 批量样本
        返回:
        -----
        np.ndarray
            增强后的数组，形状与输入一致。
        """
        # 全局跳过机制
        if np.random.rand() > self.global_prob:
            return datas

        single = False
        if datas.ndim == 2:
            # 自动添加 batch 维度
            datas = datas[np.newaxis, ...]
            single = True

        out = datas.copy()
        for func in self.augmentations:
            # 每个增强按顺序应用
            out = func(out)

        # 如果输入是单样本，去掉 batch 维度
        return out[0] if single else out
