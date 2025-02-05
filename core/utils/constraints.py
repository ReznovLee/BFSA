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
        # 每行（目标）的非零元素数量不超过1
        return np.all(assignment.getnnz(axis=1) <= 1)

    @staticmethod
    def check_radar_channels(assignment: csr_matrix, radar_network: RadarNetwork) -> bool:
        """
        检查约束式14：每个雷达的跟踪数不超过其通道数。
        :param assignment: 分配矩阵（目标×雷达），稀疏格式
        :param radar_network: 雷达网络对象
        :return: 是否满足约束
        """
        # 计算每个雷达的已分配数量
        allocated = assignment.getnnz(axis=0)
        # 检查每个雷达的分配数是否超过通道数
        for radar_id, count in enumerate(allocated):
            radar = radar_network.radars.get(radar_id)
            if radar and count > radar.channels:
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
        for i, j in zip(rows, cols):
            radar = radar_network.radars.get(j)
            target_pos = target_positions[i]
            if not radar or not radar.is_target_in_range(target_pos):
                return False
        return True

    @staticmethod
    def check_binary_variables(assignment: csr_matrix) -> bool:
        """
        检查约束式16：分配矩阵元素为0或1。
        :param assignment: 分配矩阵（目标×雷达），稀疏格式
        :return: 是否满足约束
        """
        # 稀疏矩阵的data数组应全为1（假设矩阵仅包含0和1）
        return np.all(assignment.data == 1)

    @staticmethod
    def verify_all_constraints(
        assignment: csr_matrix,
        radar_network: RadarNetwork,
        target_positions: List[np.ndarray]
    ) -> Dict[str, bool]:
        """
        综合验证所有约束，返回各约束的检查结果。
        :return: 字典形式约束检查结果，例如 {"C13": True, "C14": False, ...}
        """
        return {
            "C13": ConstraintChecker.check_single_assignment(assignment),
            "C14": ConstraintChecker.check_radar_channels(assignment, radar_network),
            "C15": ConstraintChecker.check_radar_coverage(assignment, radar_network, target_positions),
            "C16": ConstraintChecker.check_binary_variables(assignment)
        }