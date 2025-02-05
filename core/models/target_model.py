import numpy as np
from typing import Dict, List, Optional

class TargetModel:
    """
    目标运动模型基类，记录完整轨迹历史，支持按时间区间查询状态。
    """
    MACH_TO_MS = 340.0

    def __init__(
        self,
        initial_state: np.ndarray,
        dt: float = 0.1,
        speed_range: Optional[tuple] = None,
        priority: int = 1
    ):
        self.state = initial_state.astype(float)
        self.dt = dt
        self.priority = priority
        self.speed_range = (
            (speed_range[0] * self.MACH_TO_MS, speed_range[1] * self.MACH_TO_MS)
            if speed_range else None
        )
        self.trajectory = []  # 记录完整轨迹历史
        self.trajectory.append({"time": 0.0, "state": self.state.copy()})

    def update(self, current_time: float) -> np.ndarray:
        """更新目标状态，记录时间戳"""
        raise NotImplementedError

    def get_states_in_interval(self, start_time: float, end_time: float) -> List[Dict]:
        """获取指定时间区间内的所有状态"""
        return [p for p in self.trajectory if start_time <= p['time'] <= end_time]

    def _clamp_speed(self, velocity: np.ndarray) -> np.ndarray:
        """限制速度范围"""
        if self.speed_range is None:
            return velocity
        speed = np.linalg.norm(velocity)
        min_speed, max_speed = self.speed_range
        if speed < min_speed:
            return velocity * (min_speed / speed)
        elif speed > max_speed:
            return velocity * (max_speed / speed)
        return velocity

class BallisticMissile(TargetModel):
    def __init__(self, initial_state: np.ndarray, dt: float = 0.1, g: float = 9.81, speed_range: tuple = (7.5, 8.5)):
        super().__init__(initial_state, dt, speed_range, priority=1)
        self.g = g

    def update(self, current_time: float) -> np.ndarray:
        x, y, z, vx, vy, vz = self.state
        new_z = z + vz * self.dt + 0.5 * self.g * self.dt**2
        new_vz = vz + self.g * self.dt
        self.state = np.array([x + vx*self.dt, y + vy*self.dt, new_z, vx, vy, new_vz])
        self.state[3:] = self._clamp_speed(self.state[3:])
        self.trajectory.append({"time": current_time, "state": self.state.copy()})
        return self.state

class CruiseMissile(TargetModel):
    def __init__(self, initial_state: np.ndarray, dt: float = 0.1, g: float = 9.81, cruise_stage_steps: int = 70, 
                 speed_range: tuple = (0.7, 0.9), noise_std: float = 0.1, adjust_coeff: float = 0.2):
        super().__init__(initial_state, dt, speed_range, priority=2)
        self.g = g
        self.cruise_stage_steps = cruise_stage_steps
        self.total_steps = 0
        self.noise_std = noise_std
        self.adjust_coeff = adjust_coeff

    def update(self, current_time: float) -> np.ndarray:
        self.total_steps += 1
        # ...（原有逻辑，略）
        self.trajectory.append({"time": current_time, "state": self.state.copy()})
        return self.state

class FighterAircraft(TargetModel):
    def __init__(self, initial_state: np.ndarray, dt: float = 0.1, speed_range: tuple = (1.4, 1.6), 
                 noise_std: float = 0.2, z_min: float = 500):
        super().__init__(initial_state, dt, speed_range, priority=3)
        self.noise_std = noise_std
        self.z_min = z_min

    def update(self, current_time: float) -> np.ndarray:
        # ...（原有逻辑，略）
        self.trajectory.append({"time": current_time, "state": self.state.copy()})
        return self.state