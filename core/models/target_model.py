#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：BFSA
@File    ：target_model.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/2/1 9:13
"""

import numpy as np

# Mach 转换为 m/s（声速）
MACH_TO_MS = 340.0


class TargetModel:
    """ 目标模型基类（velocity_mach 为三维向量，表示各方向马赫数） """

    def __init__(self, target_id: int, position: np.ndarray, velocity_mach: np.ndarray, target_type: str, priority: int):
        """
        初始化目标
        :param target_id: 目标唯一ID
        :param position: 目标初始位置 (x, y, z)
        :param velocity_mach: 各方向马赫数向量（e.g., [1.2, 0, 0.5] 表示x方向1.2马赫，z方向0.5马赫）
        :param target_type: 目标类型
        :param priority: 目标优先级
        """
        self.target_id = target_id
        self.position = np.array(position, dtype=np.float64)
        self.velocity_mach = np.array(velocity_mach, dtype=np.float64)
        self.velocity_ms = self.velocity_mach * MACH_TO_MS  # 各方向实际速度（m/s）
        self.target_type = target_type
        self.priority = priority

    def update_position(self, time_step: float):
        """ 更新目标位置，默认匀速直线运动 """
        self.position += self.velocity_ms * time_step

    def get_state(self, timestamp: int) -> list:
        """ 获取目标当前状态（返回向量形式的马赫数） """
        return [
            self.target_id,
            timestamp,
            *self.position,
            *self.velocity_mach,  # 向量形式
            self.target_type,
            self.priority
        ]


class BallisticMissile(TargetModel):
    """ 弹道导弹类（严格物理模型，z轴向下为正重力） """

    PRIORITY = 1
    GRAVITY = np.array([0, 0, 9.81])  # 假设z轴向下为正（与位置坐标一致）

    def __init__(self, target_id: int, position: np.ndarray, velocity_mach: np.ndarray):
        super().__init__(target_id, position, velocity_mach, "ballistic_missile", self.PRIORITY)

    def update_position(self, time_step: float):
        # 更新速度：注意重力方向与坐标定义一致
        self.velocity_ms += self.GRAVITY * time_step
        self.position += self.velocity_ms * time_step
        # 更新马赫数向量（各方向独立计算）
        self.velocity_mach = self.velocity_ms / MACH_TO_MS


class CruiseMissile(TargetModel):
    """ 巡航导弹类（修正爬升逻辑，避免位置直接累加） """

    PRIORITY = 2
    CRUISE_ALTITUDE = 8000  # 巡航高度（m）

    def __init__(self, target_id: int, position: np.ndarray, velocity_mach: np.ndarray):
        super().__init__(target_id, position, velocity_mach, "cruise_missile", self.PRIORITY)
        self.current_phase = "climb"
        self._disturbance_cache = np.zeros(2)  # 缓存扰动，避免频繁计算

    def _apply_disturbance(self, time_step: float):
        """ 应用扰动 """
        self._disturbance_cache = np.random.normal(0, 0.5, 2) * time_step

    def update_position(self, time_step: float):
        if self.current_phase == "climb":
            if self.position[2] < self.CRUISE_ALTITUDE:
                # 通过修改速度实现爬升（原速度 + 垂直爬升速度）
                self.velocity_ms[2] += 50  # 爬升速率 50 m/s
            else:
                self.current_phase = "cruise"
                self.velocity_ms[2] = 0  # 停止爬升

        elif self.current_phase == "cruise":
            self._apply_disturbance(time_step)
            self.velocity_ms[:2] += self._disturbance_cache

            if np.random.rand() < 0.01:  # 1%概率进入俯冲
                self.current_phase = "dive"

        elif self.current_phase == "dive":
            self.velocity_ms[2] = -100  # 固定俯冲速度

        self.position += self.velocity_ms * time_step
        self.velocity_mach = self.velocity_ms / MACH_TO_MS


class FighterJet(TargetModel):
    """ 战斗机类（控制扰动幅度，防止速度突变） """

    PRIORITY = 3
    MIN_ALTITUDE = 500  # 最低飞行高度（m）
    MAX_DISTURBANCE = 2.0  # 扰动最大幅度（m/s²）

    def __init__(self, target_id: int, position: np.ndarray, velocity_mach: np.ndarray):
        super().__init__(target_id, position, velocity_mach, "fighter_jet", self.PRIORITY)

    def update_position(self, time_step: float):
        # 添加有限随机扰动（各方向独立）
        disturbance = np.random.normal(0, 0.5, 3)  # 标准差=0.5 m/s²
        self.velocity_ms += disturbance * time_step

        # 高度保护
        if self.position[2] < self.MIN_ALTITUDE:
            self.velocity_ms[2] = abs(self.velocity_ms[2])  # 强制上升

        self.position += self.velocity_ms * time_step
        self.velocity_mach = self.velocity_ms / MACH_TO_MS
