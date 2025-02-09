import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Union
from core.models.radar_network import RadarNetwork

class TrackingMetrics:
    """
    跟踪性能评价体系，包含多个核心指标计算：
    1. 加权总跟踪时间
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
        计算加权总跟踪时间。
        """
        tracked = assignment.sum(axis=1).A1  # 获取目标的跟踪次数（转换为 1D 数组）
        priorities = np.array([t["priority"] for t in targets])
        return np.dot(tracked, priorities) * time_step  # 计算加权总跟踪时间

    @staticmethod
    def radar_resource_utilization(
        assignment: csr_matrix,
        radar_network: RadarNetwork
    ) -> Dict[int, float]:
        """
        计算各雷达通道利用率（通道使用数 / 总通道数）。
        """
        utilization = {}
        used_channels = np.array(assignment.sum(axis=0)).flatten()
        radar_ids = list(radar_network.radars.keys())

        for i, radar_id in enumerate(radar_ids):
            radar = radar_network.radars[radar_id]
            total_channels = radar.num_channels
            utilization[radar_id] = (used_channels[i] / total_channels) * 100 if total_channels > 0 else 0

        return utilization

    @staticmethod
    def coverage_ratio(
        assignment: csr_matrix,
        num_targets: int
    ) -> float:
        """
        计算目标覆盖比例（被跟踪过的目标数 / 总目标数）。
        """
        tracked_targets = np.count_nonzero(assignment.sum(axis=1))
        return tracked_targets / num_targets

    @staticmethod
    def tracking_switches(
        assignment: csr_matrix,
        targets: List[Dict]
    ) -> Dict[int, int]:
        """
        计算每个目标的雷达切换次数。
        """
        switches = {}
        for i, target in enumerate(targets):
            row = assignment.getrow(i).toarray().ravel()
            radar_ids = np.where(row == 1)[0]

            if radar_ids.size > 1:
                switch_count = np.sum(radar_ids[:-1] != radar_ids[1:])
            else:
                switch_count = 0

            switches[target['id']] = switch_count

        return switches

    @staticmethod
    def average_tracking_delay(
        assignment: csr_matrix,
        target_entries: List[int]
    ) -> float:
        """
        计算平均跟踪延迟（目标进入范围后到首次被跟踪的时间步数）。
        """
        delays = []
        for i, entry_step in enumerate(target_entries):
            allocations = assignment.getrow(i).nonzero()[1]

            if allocations.size == 0:
                continue  # 忽略未被跟踪的目标

            first_alloc_step = allocations.min()
            delays.append(first_alloc_step - entry_step)

        return float(np.mean(delays)) if delays else 0.0

    @staticmethod
    def generate_report(
        assignment: csr_matrix,
        radar_network: RadarNetwork,
        targets: List[Dict],
        time_step: float = 1.0,
        target_entries: List[int] = None
    ) -> Dict[str, Union[float, Dict]]:
        """
        生成综合性能报告。
        """
        if target_entries is None:
            target_entries = [0] * len(targets)

        return {
            "weighted_total_time": float(TrackingMetrics.weighted_total_tracking_time(assignment, targets, time_step)),
            "resource_utilization": {int(k): float(v) for k, v in TrackingMetrics.radar_resource_utilization(assignment, radar_network).items()},
            "coverage_ratio": float(TrackingMetrics.coverage_ratio(assignment, len(targets))),
            "tracking_switches": {int(k): int(v) for k, v in TrackingMetrics.tracking_switches(assignment, targets).items()},
            "average_delay": float(TrackingMetrics.average_tracking_delay(assignment, target_entries))
        }


def priority_direction_score(target: Dict, radar_position: np.ndarray, priority_weight: float = 0.7) -> float:
    """
    计算综合优先级和方向性的评分。
    """
    target_pos = np.array(target['trajectory'][-1]['position'])
    delta_pos = radar_position - target_pos

    velocity = np.array(target['trajectory'][-1]['velocity'])
    if np.linalg.norm(velocity) < 1e-6 or np.linalg.norm(delta_pos) < 1e-6:
        direction_score = 0.5
    else:
        cos_theta = np.dot(velocity, delta_pos) / (np.linalg.norm(velocity) * np.linalg.norm(delta_pos))
        direction_score = (cos_theta + 1) / 2  # 归一化到 [0, 1]

    priority = target['priority']
    normalized_priority = (priority - 1) / 2

    score = priority_weight * normalized_priority + (1 - priority_weight) * direction_score
    return score
