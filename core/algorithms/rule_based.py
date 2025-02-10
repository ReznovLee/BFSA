import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict
from core.models.radar_network import RadarNetwork

class RuleBasedScheduler:
    """
    基于规则（就近原则）的雷达目标分配算法：
    1. 对每个目标，找到覆盖该目标且最近的雷达。
    2. 若该雷达有可用通道，则分配；否则尝试次近雷达。
    3. 若所有覆盖雷达均无通道，则该目标无法分配。
    """

    def __init__(self, radar_network: RadarNetwork):
        self.radar_network = radar_network

    def solve(self, targets: List[Dict], target_positions: List[np.ndarray]) -> csr_matrix:
        """
        生成目标-雷达分配矩阵：
        :param targets: 目标列表（包含 id 和 priority）
        :param target_positions: 目标位置列表（与 targets 顺序一致）
        :return: 稀疏分配矩阵（目标×雷达）
        """
        num_targets = len(targets)
        num_radars = len(self.radar_network.radars)
        assignment_data = []  # 存储分配数据
        assignment_rows = []  # 存储目标索引
        assignment_cols = []  # 存储雷达索引

        for i, (target, position) in enumerate(zip(targets, target_positions)):
            # 1. 查找所有覆盖目标的雷达，按距离排序
            covering_radars = self.radar_network.find_covering_radars(position)
            if not covering_radars:
                continue  # 目标未被任何雷达覆盖，跳过

            # 2. 依次尝试最近的雷达，直到成功分配
            for radar in covering_radars:
                if radar.allocate_channel(target["id"]):  # ✅ 传递目标 ID
                    assignment_rows.append(i)  # 目标索引
                    assignment_cols.append(radar.radar_id)  # 雷达索引
                    assignment_data.append(1)  # 分配值
                    break  # 成功分配后跳出循环

        # 3. 生成稀疏矩阵
        return csr_matrix((assignment_data, (assignment_rows, assignment_cols)), shape=(num_targets, num_radars), dtype=np.int8)
