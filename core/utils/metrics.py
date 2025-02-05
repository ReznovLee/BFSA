import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Union
from core.models.radar_network import RadarNetwork

class TrackingMetrics:
    """
    跟踪性能评价体系，包含多个核心指标计算。
    文档定义指标：
    1. 加权总跟踪时间（式12）
    2. 雷达资源利用率
    3. 目标覆盖比例
    4. 跟踪切换次数
    5. 平均跟踪延迟
    """

    @staticmethod
    def weighted_total_tracking_time(
        assignment: csr_matrix,
        targets: List[Dict],
        time_step: float = 1.0
    ) -> float:
        """
        计算加权总跟踪时间（目标函数式12）
        :param assignment: 分配矩阵（目标×雷达×时间步）的稀疏矩阵
        :param targets: 目标列表，含优先级权重
        :param time_step: 单个时间步的持续时间（秒）
        :return: 加权总跟踪时间（秒）
        """
        total = 0.0
        for i, target in enumerate(targets):
            # 获取目标i在所有时间步的跟踪次数
            tracked = assignment.getrow(i).sum()
            total += target['priority'] * tracked * time_step
        return total

    @staticmethod
    def radar_resource_utilization(
        assignment: csr_matrix,
        radar_network: RadarNetwork
    ) -> Dict[int, float]:
        """
        计算各雷达通道利用率（通道使用数 / 总通道数）
        :return: 字典 {雷达ID: 利用率百分比}
        """
        utilization = {}
        # 统计每个雷达的已用通道数
        used = assignment.getnnz(axis=0)
        for radar_id, radar in radar_network.radars.items():
            total_channels = radar.channels
            utilization[radar_id] = (used[radar_id] / total_channels) * 100
        return utilization

    @staticmethod
    def coverage_ratio(
        assignment: csr_matrix,
        num_targets: int
    ) -> float:
        """
        计算目标覆盖比例（被跟踪过的目标数 / 总目标数）
        """
        tracked_targets = np.unique(assignment.nonzero()[0]).size
        return tracked_targets / num_targets

    @staticmethod
    def tracking_switches(
        assignment: csr_matrix,
        targets: List[Dict]
    ) -> Dict[int, int]:
        """
        计算每个目标的雷达切换次数
        :return: 字典 {目标ID: 切换次数}
        """
        switches = {}
        for i, target in enumerate(targets):
            # 获取目标i的雷达分配序列（时间步顺序）
            row = assignment.getrow(i).toarray().ravel()
            radar_ids = np.where(row == 1)[0]
            # 计算切换次数（相邻雷达ID不同的次数）
            switch_count = np.sum(radar_ids[:-1] != radar_ids[1:])
            switches[target['id']] = switch_count
        return switches

    @staticmethod
    def average_tracking_delay(
        assignment: csr_matrix,
        target_entries: List[int]
    ) -> float:
        """
        计算平均跟踪延迟（目标进入范围后到首次被跟踪的时间步数）
        :param target_entries: 每个目标的进入时间步列表
        """
        delays = []
        for i, entry_step in enumerate(target_entries):
            # 获取目标i的分配时间序列
            allocations = assignment.getrow(i).nonzero()[1]
            if allocations.size == 0:
                continue  # 忽略未被跟踪的目标
            first_alloc_step = allocations.min()
            delays.append(first_alloc_step - entry_step)
        return np.mean(delays) if delays else 0.0

    @staticmethod
    def generate_report(
        assignment: csr_matrix,
        radar_network: RadarNetwork,
        targets: List[Dict],
        time_step: float = 1.0,
        target_entries: List[int] = None
    ) -> Dict[str, Union[float, Dict]]:
        """
        生成综合性能报告
        :param target_entries: 目标进入时间步列表（若未提供，假设从步0开始）
        """
        if target_entries is None:
            target_entries = [0] * len(targets)
        
        return {
            "weighted_total_time": TrackingMetrics.weighted_total_tracking_time(assignment, targets, time_step),
            "resource_utilization": TrackingMetrics.radar_resource_utilization(assignment, radar_network),
            "coverage_ratio": TrackingMetrics.coverage_ratio(assignment, len(targets)),
            "tracking_switches": TrackingMetrics.tracking_switches(assignment, targets),
            "average_delay": TrackingMetrics.average_tracking_delay(assignment, target_entries)
        }
    
    import numpy as np

def priority_direction_score(target: Dict, radar_position: np.ndarray, priority_weight: float = 0.7) -> float:
    """
    综合目标优先级和接近方向的评分函数。
    
    参数：
    - target: 目标字典，需包含 'priority'（优先级）和 'velocity'（速度向量）
    - radar_position: 雷达位置坐标 [x, y, z]
    - priority_weight: 优先级权重（方向权重为 1 - priority_weight）
    
    返回：
    - 评分值（范围 [0, 1]），越高表示越应分配该雷达
    """
    # 计算目标当前位置与雷达的向量差
    target_pos = np.array(target['trajectory'][-1]['position'])  # 假设取最新位置
    delta_pos = radar_position - target_pos
    
    # 方向一致性：速度向量与雷达方向的夹角余弦
    velocity = np.array(target['trajectory'][-1]['velocity'])
    if np.linalg.norm(velocity) < 1e-6 or np.linalg.norm(delta_pos) < 1e-6:
        direction_score = 0.5  # 速度或位置差过小时默认中性评分
    else:
        cos_theta = np.dot(velocity, delta_pos) / (np.linalg.norm(velocity) * np.linalg.norm(delta_pos))
        direction_score = (cos_theta + 1) / 2  # 归一化到 [0, 1]
    
    # 优先级归一化（假设优先级范围为 1-3）
    priority = target['priority']
    normalized_priority = (priority - 1) / 2  # 1→0, 2→0.5, 3→1
    
    # 加权综合评分
    score = priority_weight * normalized_priority + (1 - priority_weight) * direction_score
    return score