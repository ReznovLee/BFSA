import numpy as np
from typing import Dict, List, Union
from scipy.sparse import csr_matrix
from core.models.radar_network import RadarNetwork

class ConstraintChecker:
    """
    约束检查工具类，用于验证雷达网络调度中的硬性约束。
    文档定义约束：
    1. 每个目标同一时刻只能被一个雷达跟踪（约束式13）。
    2. 每个雷达的跟踪数量不能超过其通道数（约束式14）。
    3. 目标必须在雷达的覆盖范围内才能被分配（约束式15）。
    4. 分配变量为0-1二元变量（约束式16）。
    """

    @staticmethod
    def check_single_assignment(assignment: csr_matrix) -> bool:
        """
        检查约束式13：每个目标同一时刻只能被一个雷达跟踪。
        :param assignment: 分配矩阵（目标×雷达），稀疏格式
        :return: 是否满足约束
        """
        return np.all(assignment.sum(axis=1) <= 1)  # 目标最多被1个雷达分配

    @staticmethod
    def check_radar_channels(assignment: csr_matrix, radar_network: RadarNetwork) -> bool:
        """
        检查约束式14：每个雷达的跟踪数不超过其通道数。
        :param assignment: 分配矩阵（目标×雷达），稀疏格式
        :param radar_network: 雷达网络对象
        :return: 是否满足约束
        """
        # 计算每个雷达的已分配目标数
        allocated_targets = np.array(assignment.sum(axis=0)).flatten()
        radar_ids = list(radar_network.radars.keys())

        for i, radar_id in enumerate(radar_ids):
            radar = radar_network.radars[radar_id]
            if allocated_targets[i] > radar.num_channels:  # 确保不超过雷达通道数
                return False
        return True

    @staticmethod
    def check_radar_coverage(
        assignment: csr_matrix,
        radar_network: RadarNetwork,
        target_positions: List[np.ndarray]
    ) -> bool:
        """
        检查约束式15：所有分配的目标必须在雷达覆盖范围内。
        :param assignment: 分配矩阵（目标×雷达），稀疏格式
        :param radar_network: 雷达网络对象
        :param target_positions: 目标位置列表（与矩阵行索引对应）
        :return: 是否满足约束
        """
        rows, cols = assignment.nonzero()
        for target_idx, radar_idx in zip(rows, cols):
            if target_idx >= len(target_positions):  # 防止索引越界
                return False
            radar = radar_network.radars.get(radar_idx)
            if radar is None or not radar.is_target_in_range(target_positions[target_idx]):
                return False
        return True

    @staticmethod
    def check_binary_variables(assignment: csr_matrix) -> bool:
        """
        检查约束式16：分配矩阵元素为0或1。
        :param assignment: 分配矩阵（目标×雷达），稀疏格式
        :return: 是否满足约束
        """
        return np.all(np.isin(assignment.data, [0, 1]))  # 仅包含0或1

    @staticmethod
    def verify_all_constraints(
        assignment: csr_matrix,
        radar_network: RadarNetwork,
        target_positions: List[np.ndarray]
    ) -> Dict[str, Union[bool, List[int]]]:
        """
        综合验证所有约束，返回各约束的检查结果。
        :return: 字典形式约束检查结果，例如:
                 {"C13": True, "C14": [超载雷达ID], "C15": [目标超出覆盖范围的ID], "C16": True}
        """
        results = {
            "C13": ConstraintChecker.check_single_assignment(assignment),
            "C14": [],
            "C15": [],
            "C16": ConstraintChecker.check_binary_variables(assignment)
        }

        # 检查雷达通道数约束
        allocated_targets = np.array(assignment.sum(axis=0)).flatten()
        radar_ids = list(radar_network.radars.keys())

        for i, radar_id in enumerate(radar_ids):
            radar = radar_network.radars[radar_id]
            if allocated_targets[i] > radar.num_channels:
                results["C14"].append(radar_id)  # 记录超载的雷达 ID

        # 检查目标覆盖约束
        rows, cols = assignment.nonzero()
        for target_idx, radar_idx in zip(rows, cols):
            if target_idx >= len(target_positions):  # 防止索引越界
                results["C15"].append(target_idx)
            else:
                radar = radar_network.radars.get(radar_idx)
                if radar is None or not radar.is_target_in_range(target_positions[target_idx]):
                    results["C15"].append(target_idx)  # 记录超出范围的目标ID

        return results
