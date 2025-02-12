import numpy as np
from typing import Optional


class KalmanFilter:
    """
    卡尔曼滤波器，用于目标状态估计和预测。
    目标状态向量: [x, y, z, vx, vy, vz]
    默认模型: 匀速运动 (CV)，可扩展为加速度或其他模型。
    """

    def __init__(
            self,
            dim_state: int = 6,
            dim_obs: int = 3,
            dt: float = 1.0,
            process_noise_std: float = 0.1,
            obs_noise_std: float = 0.5,
            F: Optional[np.ndarray] = None,
            H: Optional[np.ndarray] = None,
            Q: Optional[np.ndarray] = None,
            R: Optional[np.ndarray] = None
    ):
        """
        初始化卡尔曼滤波器
        :param dim_state: 状态维度 (默认 6: x, y, z, vx, vy, vz)
        :param dim_obs: 观测维度 (默认 3: x, y, z)
        :param dt: 时间步长
        :param process_noise_std: 过程噪声标准差
        :param obs_noise_std: 观测噪声标准差
        :param F: 状态转移矩阵 (可选自定义)
        :param H: 观测矩阵 (可选自定义)
        :param Q: 过程噪声协方差矩阵 (可选自定义)
        :param R: 观测噪声协方差矩阵 (可选自定义)
        """
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.dt = dt

        # 初始化状态向量和协方差矩阵
        self.x = np.zeros(dim_state)
        self.P = np.eye(dim_state) * 1e2

        # 状态转移矩阵（匀速运动模型）
        self.F = F if F is not None else np.eye(dim_state)
        self.F[:3, 3:] = np.eye(3) * dt  # 位置 += 速度 * dt
        assert self.F.shape == (dim_state, dim_state), "F 维度错误"

        # 观测矩阵（默认仅观测位置）
        self.H = H if H is not None else np.zeros((dim_obs, dim_state))
        self.H[:3, :3] = np.eye(3)
        assert self.H.shape == (dim_obs, dim_state), "H 维度错误"

        # 过程噪声协方差矩阵
        self.Q = Q if Q is not None else np.eye(dim_state) * (process_noise_std ** 2)
        self.Q[3:, 3:] *= 0.1  # 速度噪声较小
        assert self.Q.shape == (dim_state, dim_state), "Q 维度错误"

        # 观测噪声协方差矩阵
        self.R = R if R is not None else np.eye(dim_obs) * (obs_noise_std ** 2)
        assert self.R.shape == (dim_obs, dim_obs), "R 维度错误"

    def initialize(self, initial_state: np.ndarray) -> None:
        """初始化滤波器状态"""
        assert initial_state.shape == (self.dim_state,), f"初始状态必须是 {self.dim_state} 维"
        self.x = initial_state.astype(float)
        self.P = np.eye(self.dim_state) * 1e2

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """重置滤波器到初始状态"""
        if initial_state is None:
            initial_state = np.zeros(self.dim_state)
        self.initialize(initial_state)

    def predict(self) -> np.ndarray:
        """预测下一时刻状态"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        根据观测值更新状态估计
        :param z: 观测向量 [x, y, z] 或 [x, y, z, vx, vy, vz]
        """
        assert z.shape[0] == self.dim_obs, f"观测值维度应为 {self.dim_obs}, 当前为 {z.shape}"

        # 计算卡尔曼增益
        y = z - self.H @ self.x  # 观测残差
        S = self.H @ self.P @ self.H.T + self.R  # 观测不确定性
        S += np.eye(self.dim_obs) * 1e-6  # 防止 S 变成奇异矩阵
        K = self.P @ self.H.T @ np.linalg.pinv(S)  # 使用伪逆，避免求逆错误

        # 更新状态和协方差（Joseph 形式，数值稳定）
        self.x += K @ y
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def get_state(self) -> np.ndarray:
        """返回当前状态估计值"""
        return self.x.copy()

    def get_position(self) -> np.ndarray:
        """返回估计的位置 [x, y, z]"""
        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """返回估计的速度 [vx, vy, vz]"""
        return self.x[3:].copy()
