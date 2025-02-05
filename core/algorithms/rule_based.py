import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Optional
from core.models.radar_network import RadarNetwork

class RuleBasedScheduler:
    """
    基于规则的雷达目标分配算法（就近原则）。
    分配逻辑：
    1. 对每个目标，找到覆盖该目标且距离最近的雷达。
    2. 若该雷达有可用通道，则分配；否则尝试次近雷达。
    3. 若所有覆盖雷达均无通道，则无法分配。
    """

    def __init__(self, radar_network: RadarNetwork):
        self.radar_network = radar_network

    def solve(self, targets: List[Dict], target_positions: List[np.ndarray]) -> csr_matrix:
        """
        生成分配矩阵（目标×雷达），元素为1表示分配。
        :param targets: 目标列表，每个目标包含id和优先级
        :param target_positions: 目标位置列表（与targets顺序一致）
        :return: 稀疏分配矩阵
        """
        num_targets = len(targets)
        num_radars = len(self.radar_network.radars)
        assignment = csr_matrix((num_targets, num_radars), dtype=np.int8)

        for i, (target, position) in enumerate(zip(targets, target_positions)):
            # 1. 找到覆盖目标的所有雷达并按距离排序
            covering_radars = self.radar_network.find_covering_radars(position)
            if not covering_radars:
                continue  # 目标未被任何雷达覆盖

            # 2. 按距离从近到远尝试分配
            for radar in covering_radars:
                if radar.allocate_channel():
                    assignment[i, radar.id] = 1
                    break  # 分配成功，跳出循环

        return assignment

    @staticmethod
    def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算三维欧氏距离"""
        return np.linalg.norm(pos1 - pos2)