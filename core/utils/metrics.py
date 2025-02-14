#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：BFSA
@File    ：metrics.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/2/9 11:03
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Union
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
    def weighted_total_tracking_time(assignment: csr_matrix, targets: List[Dict], time_step: float = 1.0) -> float:
        """计算加权总跟踪时间"""
        if assignment.shape[0] == 0:
            return 0

        tracked = assignment.getnnz(axis=1)  # 获取每个目标的跟踪次数
        priorities = np.array([t["priority"] for t in targets])

        # 确保优先级数组与目标数目一致
        full_priorities = np.zeros(assignment.shape[0])
        tracked_indices = np.unique(assignment.nonzero()[0])
        full_priorities[tracked_indices] = priorities[tracked_indices]

        return np.dot(tracked, full_priorities) * time_step

    @staticmethod
    def radar_resource_utilization(assignment: csr_matrix, radar_network: RadarNetwork) -> Dict[int, float]:
        """计算各雷达通道利用率"""
        utilization = {}
        used_channels = np.array(assignment.sum(axis=0)).flatten()
        radar_ids = list(radar_network.radars.keys())

        for i, radar_id in enumerate(radar_ids):
            if i < len(used_channels):
                radar = radar_network.radars[radar_id]
                total_channels = radar.num_channels
                utilization[radar_id] = (used_channels[i] / total_channels) * 100 if total_channels > 0 else 0

        return utilization

    @staticmethod
    def coverage_ratio(assignment: csr_matrix, num_targets: int) -> float:
        """计算目标覆盖比例"""
        if num_targets == 0:
            return 0.0
        tracked_targets = np.count_nonzero(np.any(assignment.toarray(), axis=1))  # 计算被跟踪的目标数
        return tracked_targets / num_targets

    @staticmethod
    def tracking_switches(assignment: csr_matrix, targets: List[Dict]) -> Dict[int, int]:
        """计算每个目标的雷达切换次数"""
        switches = {}
        num_targets = min(assignment.shape[0], len(targets))

        # 遍历每个目标，计算其雷达切换次数
        for i in range(num_targets):
            if i >= assignment.shape[0]:
                continue

            row = assignment.getrow(i).toarray().ravel()
            radar_ids = np.where(row == 1)[0]

            if radar_ids.size > 1:
                switch_count = np.sum(radar_ids[:-1] != radar_ids[1:])
            else:
                switch_count = 0

            switches[targets[i]["id"]] = switch_count

        return switches

    @staticmethod
    def average_tracking_delay(assignment: csr_matrix, target_entries: List[int]) -> float:
        """计算平均跟踪延迟"""
        delays = []
        num_targets = min(assignment.shape[0], len(target_entries))

        for i in range(num_targets):
            if i >= assignment.shape[0]:
                continue

            allocations = assignment.getrow(i).nonzero()[1]
            if allocations.size == 0:
                continue

            first_alloc_step = allocations.min()
            delays.append(first_alloc_step - target_entries[i])

        return float(np.mean(delays)) if delays else 0.0

    @staticmethod
    def generate_report(assignment: csr_matrix, radar_network: RadarNetwork, targets: List[Dict], time_step: float = 1.0, target_entries: List[int] = None) -> Dict[str, Union[float, Dict]]:
        """生成综合性能报告"""
        if target_entries is None:
            target_entries = [0] * len(targets)

        report = {
            "weighted_total_time": TrackingMetrics.weighted_total_tracking_time(assignment, targets, time_step),
            "resource_utilization": TrackingMetrics.radar_resource_utilization(assignment, radar_network),
            "coverage_ratio": TrackingMetrics.coverage_ratio(assignment, len(targets)),
            "tracking_switches": TrackingMetrics.tracking_switches(assignment, targets),
            "average_delay": TrackingMetrics.average_tracking_delay(assignment, target_entries)
        }

        # 转换结果中的字典为浮点数值，确保没有数据类型问题
        report["resource_utilization"] = {int(k): float(v) for k, v in report["resource_utilization"].items()}
        return report


def priority_direction_score(target: Dict, radar_position: np.ndarray, priority_weight: float = 0.7) -> float:
    """计算综合优先级和方向性评分"""
    target_pos = np.array(target.get("trajectory", [{"position": np.zeros(3)}])[-1]["position"])
    delta_pos = radar_position - target_pos

    velocity = np.array(target.get("trajectory", [{"velocity": np.zeros(3)}])[-1]["velocity"])
    if np.linalg.norm(velocity) < 1e-6 or np.linalg.norm(delta_pos) < 1e-6:
        direction_score = 0.5
    else:
        cos_theta = np.dot(velocity, delta_pos) / (np.linalg.norm(velocity) * np.linalg.norm(delta_pos))
        direction_score = (cos_theta + 1) / 2

    priority = target["priority"]
    normalized_priority = (priority - 1) / 2

    return priority_weight * normalized_priority + (1 - priority_weight) * direction_score
