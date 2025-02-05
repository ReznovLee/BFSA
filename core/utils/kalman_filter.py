import numpy as np

class KalmanFilter:
    """
    卡尔曼滤波器，用于目标状态估计和预测。
    文档定义：目标状态向量为 [x, y, z, vx, vy, vz]。
    默认假设为匀速运动模型，支持用户自定义动态模型参数。
    """
    
    def __init__(
        self,
        dim_state: int = 6,
        dim_obs: int = 3,
        dt: float = 1.0,
        process_noise_std: float = 0.1,
        obs_noise_std: float = 0.5
    ):
        """
        :param dim_state: 状态维度（默认6维：x, y, z, vx, vy, vz）
        :param dim_obs: 观测维度（默认3维：x, y, z）
        :param dt: 时间步长（秒）
        :param process_noise_std: 过程噪声标准差（控制模型不确定性）
        :param obs_noise_std: 观测噪声标准差（控制传感器误差）
        """
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.dt = dt
        
        # 状态向量和协方差矩阵初始化
        self.x = np.zeros(dim_state)        # 初始状态：[x, y, z, vx, vy, vz]
        self.P = np.eye(dim_state)          # 初始协方差矩阵
        
        # 状态转移矩阵（匀速模型）
        self.F = np.eye(dim_state)
        self.F[:3, 3:] = np.eye(3) * dt     # 位置 += 速度*dt
        
        # 观测矩阵（仅观测位置）
        self.H = np.zeros((dim_obs, dim_state))
        self.H[:3, :3] = np.eye(3)
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(dim_state) * (process_noise_std ** 2)
        self.Q[3:, 3:] *= 0.1  # 假设速度噪声较小
        
        # 观测噪声协方差矩阵
        self.R = np.eye(dim_obs) * (obs_noise_std ** 2)
        
    def initialize(self, initial_state: np.ndarray) -> None:
        """初始化滤波器状态"""
        self.x = initial_state.astype(float)
        self.P = np.eye(self.dim_state)  # 初始不确定性较高
        
    def predict(self) -> np.ndarray:
        """预测下一时刻状态"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        根据观测值更新状态估计
        :param z: 观测向量 [x, y, z]
        """
        if z.shape != (self.dim_obs,):
            raise ValueError(f"观测值维度应为({self.dim_obs},)，当前为{z.shape}")
        
        # 计算卡尔曼增益
        y = z - self.H @ self.x                 # 观测残差
        S = self.H @ self.P @ self.H.T + self.R  # 残差协方差
        K = self.P @ self.H.T @ np.linalg.inv(S) # 卡尔曼增益
        
        # 更新状态和协方差
        self.x += K @ y
        self.P = (np.eye(self.dim_state) - K @ self.H) @ self.P
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