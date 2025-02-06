import numpy as np

# Mach 转换为 m/s
MACH_TO_MS = 340.0

class TargetModel:
    """ 目标模型基类 """

    def __init__(self, target_id, position, velocity_mach, target_type, priority):
        """
        初始化目标
        :param target_id: 目标唯一ID
        :param position: 目标初始位置 (x, y, z)
        :param velocity_mach: 目标速度 (Mach)
        :param target_type: 目标类型 (ballistic_missile, cruise_missile, fighter_jet)
        :param priority: 目标优先级
        """
        self.target_id = target_id
        self.position = np.array(position, dtype=np.float64)
        self.velocity_mach = np.array(velocity_mach, dtype=np.float64)
        self.velocity_ms = self.velocity_mach * MACH_TO_MS  # Mach 转换为 m/s
        self.target_type = target_type
        self.priority = priority

    def update_position(self, time_step):
        """ 更新目标位置，默认匀速直线运动 """
        self.position += self.velocity_ms * time_step

    def get_state(self, timestamp):
        """ 获取目标当前状态 """
        return [self.target_id, timestamp, *self.position, *self.velocity_mach, self.target_type, self.priority]


class BallisticMissile(TargetModel):
    """ 弹道导弹类，受重力影响 """

    GRAVITY = np.array([0, 0, -9.81])
    PRIORITY = 1  # 优先级最高

    def __init__(self, target_id, position, velocity_mach):
        super().__init__(target_id, position, velocity_mach, "ballistic_missile", self.PRIORITY)

    def update_position(self, time_step):
        """ 弹道导弹受重力影响 """
        self.velocity_ms += self.GRAVITY * time_step
        self.position += self.velocity_ms * time_step
        self.velocity_mach = self.velocity_ms / MACH_TO_MS  # 转回 Mach


class CruiseMissile(TargetModel):
    """ 巡航导弹类，具有爬升、巡航、俯冲阶段 """

    PRIORITY = 2  # 中等优先级
    CRUISE_ALTITUDE = 8000  # 预设巡航高度

    def __init__(self, target_id, position, velocity_mach):
        super().__init__(target_id, position, velocity_mach, "cruise_missile", self.PRIORITY)
        self.current_phase = "climb"

    def update_position(self, time_step):
        """ 巡航导弹运动模型 """
        if self.current_phase == "climb":
            if self.position[2] < self.CRUISE_ALTITUDE:
                self.position[2] += 50 * time_step
            else:
                self.current_phase = "cruise"

        elif self.current_phase == "cruise":
            disturbance = np.random.normal(0, 1, 3)
            self.velocity_ms += disturbance * 0.1
            if np.random.rand() < 0.05:
                self.current_phase = "dive"

        elif self.current_phase == "dive":
            self.position[2] -= 100 * time_step

        self.position += self.velocity_ms * time_step
        self.velocity_mach = self.velocity_ms / MACH_TO_MS


class FighterJet(TargetModel):
    """ 战斗机类，高机动性 """

    PRIORITY = 3  # 优先级最低
    MIN_ALTITUDE = 500  # 预设最低飞行高度

    def __init__(self, target_id, position, velocity_mach):
        super().__init__(target_id, position, velocity_mach, "fighter_jet", self.PRIORITY)

    def update_position(self, time_step):
        """ 战斗机运动模型，模拟机动性 """
        disturbance = np.random.normal(0, 5, 3)
        self.velocity_ms += disturbance

        if self.position[2] + self.velocity_ms[2] * time_step < self.MIN_ALTITUDE:
            self.velocity_ms[2] = abs(self.velocity_ms[2])

        self.position += self.velocity_ms * time_step
        self.velocity_mach = self.velocity_ms / MACH_TO_MS
