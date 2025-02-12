import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict
from core.models.radar_network import RadarNetwork
from core.utils.kalman_filter import KalmanFilter
from core.utils.metrics import priority_direction_score
import logging


class BFSARHO:
    """
    基于滚动时域优化的后向-前向调度算法（BFSA-RHO），核心逻辑：
    1. Backward Stage：基于历史分配加权生成初始解。
    2. Forward Stage：使用卡尔曼滤波预测未来状态调整分配。
    3. Fusion Stage：结合优先级和方向一致性进行最终分配修正。
    """

    def __init__(self, radar_network: RadarNetwork, alpha: float = 0.9, gamma: float = 0.8, window_size: int = 3, prediction_steps: int = 2):
        self.radar_network = radar_network
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.history = []  # 记录历史分配矩阵（稀疏格式）
        self.trackers = {}  # 目标 ID 映射到卡尔曼滤波器
        self.cached_covering_radars = {}

    def solve(self, targets: List[Dict], target_positions: List[np.ndarray]) -> csr_matrix:
        """ 计算当前时刻的目标-雷达分配矩阵 """
        self._update_parameters(targets)
        assignment_backward = self._backward_stage()
        assignment_forward = self._forward_stage(assignment_backward, targets, target_positions)
        final_assignment = self._fusion_stage(assignment_forward, targets, target_positions)
        self._update_history(final_assignment)
        return final_assignment

    def _update_parameters(self, targets: List[Dict]) -> None:
        """ 根据目标平均速度动态调整 alpha 和 gamma """
        avg_speed = np.mean([np.linalg.norm(t.get('velocity', np.zeros(3))) for t in targets])
        self.alpha = max(0.5, 0.9 - 0.1 * avg_speed / 1000)
        self.gamma = max(0.6, 0.8 - 0.05 * avg_speed / 1000)

    def _backward_stage(self) -> csr_matrix:
        """ 基于时间窗口的历史加权生成初始分配 """
        if not self.history:
            return self._random_assignment()

        valid_history = self.history[-self.window_size:]
        weights = np.array([self.alpha ** (len(self.history) - t) for t in range(len(valid_history))])
        weights /= weights.sum()  # 归一化权重，防止溢出

        weighted_matrix = sum(A.multiply(w) for A, w in zip(valid_history, weights))

        row_sum = weighted_matrix.sum(axis=1).A.ravel()
        row_sum[row_sum == 0] = 1
        normalized = weighted_matrix.multiply(1 / row_sum[:, None])

        assignment = csr_matrix(weighted_matrix.shape, dtype=np.int8)
        for i in range(normalized.shape[0]):
            row = normalized.getrow(i).toarray().ravel()
            if np.any(row > 0):
                j = np.random.choice(len(row), p=row)
                assignment[i, j] = 1
        return assignment

    def _forward_stage(self, assignment_backward: csr_matrix, targets: List[Dict], target_positions: List[np.ndarray]) -> csr_matrix:
        """ 结合卡尔曼滤波预测未来状态，调整初始分配 """
        assignment_forward = assignment_backward.copy()

        for i, pos in enumerate(target_positions):
            if i >= assignment_backward.shape[0]:
                continue
            current_radar = assignment_backward.getrow(i).nonzero()[1]

            if current_radar.size == 0:
                continue

            # 初始化或更新卡尔曼滤波器
            if i not in self.trackers:
                self.trackers[i] = KalmanFilter()
                self.trackers[i].initialize(np.concatenate([pos, np.zeros(3)]))
            else:
                self.trackers[i].predict()
                self.trackers[i].update(pos)

            pred_positions = [self.trackers[i].predict()[:3] for _ in range(self.prediction_steps)]
            future_radars = set(r for pred_pos in pred_positions for r in self._get_covering_radars(pred_pos))

            if current_radar[0] not in future_radars:
                assignment_forward[i, current_radar[0]] = 0

        return assignment_forward

    def _fusion_stage(self, assignment_forward: csr_matrix, targets: List[Dict], target_positions: List[np.ndarray]) -> csr_matrix:
        """ 结合目标优先级和方向一致性填补未分配目标 """
        final_assignment = assignment_forward.copy()

        for i, target in enumerate(targets):
            if i >= final_assignment.shape[0]:
                continue

            if final_assignment.getrow(i).nnz == 0:
                candidate_radars = self.radar_network.find_covering_radars(target_positions[i])
                if candidate_radars:
                    best_radar = max(candidate_radars, key=lambda r: priority_direction_score(target, r.position))
                    channel_id = best_radar.allocate_channel(target["id"])
                    if channel_id is not None:
                        final_assignment[i, best_radar.radar_id] = 1

        return final_assignment

    def _random_assignment(self) -> csr_matrix:
        """ 无历史数据时，随机初始化分配 """
        num_targets = len(self.radar_network.radars)
        assignment = csr_matrix((num_targets, len(self.radar_network.radars)), dtype=np.int8)

        available_radars = [j for j, r in self.radar_network.radars.items() if self.radar_network.is_radar_available(j)]

        if available_radars:
            target_indices = np.arange(num_targets)
            chosen_radars = np.random.choice(available_radars, size=num_targets, replace=True)
            assignment[target_indices, chosen_radars] = 1

        return assignment

    def _update_history(self, assignment: csr_matrix) -> None:
        """ 更新历史记录队列 """
        if len(self.history) >= self.window_size:
            self.history.pop(0)
        self.history.append(assignment.copy())

    def _get_covering_radars(self, target_position: np.ndarray):
        """ 缓存雷达覆盖区域 """
        if tuple(target_position) not in self.cached_covering_radars:
            self.cached_covering_radars[tuple(target_position)] = self.radar_network.find_covering_radars(target_position)
        return self.cached_covering_radars[tuple(target_position)]
