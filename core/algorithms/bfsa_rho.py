import numpy as np
from scipy.sparse import csr_matrix, vstack
from typing import List, Dict, Optional
from core.models.radar_network import RadarNetwork
from core.models.target_model import TargetModel
from core.utils.kalman_filter import KalmanFilter
from core.utils.metrics import priority_direction_score
from core.utils.constraints import ConstraintChecker

class BFSARHO:
    """
    基于滚动时域优化的后向-前向调度算法（BFSA-RHO），核心逻辑：
    1. Backward Stage：基于历史分配加权生成初始解。
    2. Forward Stage：预测未来状态调整分配。
    3. Fusion Stage：结合优先级和方向一致性填补未分配目标。
    """

    def __init__(
        self,
        radar_network: RadarNetwork,
        alpha: float = 0.9,
        gamma: float = 0.8,
        window_size: int = 3,
        prediction_steps: int = 2
    ):
        """
        :param radar_network: 雷达网络对象（需提前初始化）
        :param alpha: 历史衰减系数（默认0.9）
        :param gamma: 预测衰减系数（默认0.8）
        :param window_size: 历史窗口大小（保留最近N个分配）
        :param prediction_steps: 前向预测步数（默认2步）
        """
        self.radar_network = radar_network
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.history = []  # 存储历史分配矩阵（稀疏格式）
        self.trackers = {}  # 目标ID到卡尔曼滤波器的映射

    def solve(
        self,
        targets: List[Dict],
        target_positions: List[np.ndarray]
    ) -> csr_matrix:
        """
        主入口：生成当前时刻的雷达-目标分配矩阵
        :param targets: 目标列表（需包含id和priority）
        :param target_positions: 目标当前位置（与targets顺序一致）
        :return: 稀疏分配矩阵（目标×雷达）
        """
        # 0. 动态调整参数（可选）
        self._update_parameters(targets)

        # 1. Backward阶段：基于历史生成初始分配
        assignment_backward = self._backward_stage()

        # 2. Forward阶段：预测未来状态调整分配
        assignment_forward = self._forward_stage(assignment_backward, target_positions)

        # 3. Fusion阶段：填补未分配目标
        final_assignment = self._fusion_stage(assignment_forward, targets, target_positions)

        # 4. 更新历史记录
        self._update_history(final_assignment)

        return final_assignment

    def _update_parameters(self, targets: List[Dict]) -> None:
        """动态调整alpha和gamma（基于目标平均速度）"""
        # avg_speed = np.mean([np.linalg.norm(t['velocity']) for t in targets])
        avg_speed = np.mean([np.linalg.norm(t['trajectory'][-1]['velocity']) for t in targets])
        self.alpha = max(0.5, 0.9 - 0.1 * avg_speed / 1000)  # 速度越快，历史权重越低
        self.gamma = max(0.6, 0.8 - 0.05 * avg_speed / 1000) # 速度越快，预测权重越低

    def _backward_stage(self) -> csr_matrix:
        """基于时间窗口的历史加权生成初始分配"""
        if not self.history:
            return self._random_assignment()

        # 仅保留最近window_size个历史分配
        valid_history = self.history[-self.window_size:]
        weighted_matrix = sum(
            [self.alpha ** (len(self.history) - t) * A for t, A in enumerate(valid_history)]
        )

        # 归一化并生成概率分布
        row_sum = weighted_matrix.sum(axis=1).A.ravel()
        row_sum[row_sum == 0] = 1  # 避免除零
        normalized = weighted_matrix.multiply(1 / row_sum[:, None])

        # 采样生成分配矩阵
        assignment = csr_matrix(weighted_matrix.shape, dtype=np.int8)
        for i in range(normalized.shape[0]):
            row = normalized.getrow(i).toarray().ravel()
            if np.any(row > 0):
                j = np.random.choice(len(row), p=row)
                assignment[i, j] = 1
        return assignment

    def _forward_stage(
        self,
        assignment_backward: csr_matrix,
        target_positions: List[np.ndarray]
    ) -> csr_matrix:
        """结合卡尔曼滤波预测未来状态，调整初始分配"""
        assignment_forward = assignment_backward.copy()
        for i, pos in enumerate(target_positions):
            # 获取或初始化卡尔曼滤波器
            tracker = self.trackers.get(i)
            if tracker is None:
                tracker = KalmanFilter()
                tracker.initialize(np.concatenate([pos, np.zeros(3)]))  # 初始速度设为0
                self.trackers[i] = tracker
            else:
                tracker.predict()
                tracker.update(pos)  # 用当前位置更新滤波器

            # 预测未来状态并检查覆盖
            future_radars = []
            for step in range(1, self.prediction_steps + 1):
                pred_state = tracker.predict()
                pred_pos = pred_state[:3]
                # 查找未来可能覆盖的雷达
                future_radars.extend(self.radar_network.find_covering_radars(pred_pos))

            # 若当前分配的雷达在未来不可用，则取消分配
            current_radar = assignment_backward.getrow(i).nonzero()[1]
            if current_radar.size > 0 and current_radar[0] not in future_radars:
                assignment_forward[i, current_radar[0]] = 0

        return assignment_forward

    def _forward_stage(
        self,
        assignment_backward: csr_matrix,
        target_positions: List[np.ndarray]
    ) -> csr_matrix:
        """结合卡尔曼滤波预测未来状态，调整初始分配"""
        assignment_forward = assignment_backward.copy()
        num_radars = assignment_backward.shape[1]  # 雷达总数（矩阵列数）

        for i, pos in enumerate(target_positions):
            # 获取或初始化卡尔曼滤波器
            tracker = self.trackers.get(i)
            if tracker is None:
                tracker = KalmanFilter()
                tracker.initialize(np.concatenate([pos, np.zeros(3)]))  # 初始速度设为0
                self.trackers[i] = tracker
            else:
                tracker.predict()
                tracker.update(pos)  # 用当前位置更新滤波器

            # 预测未来状态并检查覆盖
            future_radars = []
            for step in range(1, self.prediction_steps + 1):
                pred_state = tracker.predict()
                pred_pos = pred_state[:3]
                # 查找未来可能覆盖的雷达（返回有效的矩阵列索引）
                covering_radars = self.radar_network.find_covering_radars(pred_pos)
                future_radars.extend(covering_radars)

            # 检查当前分配的雷达是否在未来覆盖列表中
            current_radar_row = assignment_backward.getrow(i)
            if current_radar_row.nnz > 0:
                current_radar = current_radar_row.nonzero()[1][0]
                # 确保雷达索引严格有效
                if current_radar < num_radars:
                    if current_radar not in future_radars:
                        assignment_forward[i, current_radar] = 0
                else:
                    # 处理无效索引（例如日志记录或抛出异常）
                    raise ValueError(f"雷达索引 {current_radar} 超出范围 [0, {num_radars-1}]")

        return assignment_forward

    def _random_assignment(self) -> csr_matrix:
        """随机初始分配（仅用于无历史数据时）"""
        num_targets = len(self.radar_network.radars)  # 假设目标数等于雷达数（需根据实际情况调整）
        assignment = csr_matrix((num_targets, len(self.radar_network.radars)), dtype=np.int8)
        for i in range(num_targets):
            # available_radars = [j for j, r in self.radar_network.radars.items() if r.is_available()]
            available_radars = [
                j for j, r in self.radar_network.radars.items()
                if self.radar_network.is_radar_available(j)  # 使用雷达ID调用方法
            ]
            if available_radars:
                j = np.random.choice(available_radars)
                assignment[i, j] = 1
        return assignment

    def _update_history(self, assignment: csr_matrix) -> None:
        """更新历史记录队列"""
        if len(self.history) >= self.window_size:
            self.history.pop(0)
        self.history.append(assignment.copy())