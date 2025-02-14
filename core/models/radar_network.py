#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：BFSA
@File    ：radar_network.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/2/1 19:43
"""

from typing import Dict, Optional
import numpy as np
import logging


class Radar:
    """ 单部雷达类（支持半球探测范围） """

    def __init__(self, radar_id: int, position: np.ndarray, num_channels: int, detection_range: float):
        """
        初始化雷达
        :param radar_id: 雷达唯一ID
        :param position: 雷达位置 (x, y, z)，假设z坐标为雷达高度（地面以上）
        :param num_channels: 雷达的通道数
        :param detection_range: 雷达的探测范围 (m)
        """
        assert num_channels > 0, "雷达通道数必须大于0"
        assert detection_range > 0, "探测范围必须大于0"

        self.radar_id = radar_id
        self.position = np.array(position, dtype=np.float64)
        self.num_channels = num_channels
        self.detection_range = detection_range
        self.channels = {i: None for i in range(num_channels)}  # 通道状态

    def is_available(self) -> bool:
        """ 检查是否有空闲通道 """
        return any(channel is None for channel in self.channels.values())

    def allocate_channel(self, target_id: int) -> Optional[int]:
        """ 为目标分配空闲通道 """
        for channel_id in range(self.num_channels):
            if self.channels[channel_id] is None:
                self.channels[channel_id] = target_id
                logging.debug(f"Radar {self.radar_id} 分配通道 {channel_id} 给目标 {target_id}")
                return channel_id
        return None  # 无可用通道

    def release_channel(self, channel_id: int):
        """ 释放通道 """
        if self.channels[channel_id] is not None:
            released_target = self.channels[channel_id]
            self.channels[channel_id] = None
            logging.debug(f"Radar {self.radar_id} 释放通道 {channel_id}（原目标 {released_target}）")

    def is_target_in_range(self, target_position: np.ndarray) -> bool:
        """
        判断目标是否在雷达的探测范围内（上半球）
        :param target_position: 目标位置 (x, y, z)
        :return: 是否在探测范围内
        """
        target_pos = np.array(target_position)

        # 检查目标是否在雷达上方（假设雷达位于地面）
        if target_pos[2] < self.position[2]:
            return False  # 目标低于雷达高度（地面以下不探测）

        # 计算三维欧氏距离
        distance = np.linalg.norm(self.position - target_pos)

        # Debug: 输出目标与雷达之间的距离
        logging.debug(f"雷达 {self.radar_id} 与目标位置 {target_position} 的距离为 {distance:.2f}m")

        # 返回是否在探测范围内
        return distance <= self.detection_range


class RadarNetwork:
    """ 雷达网络类 """

    def __init__(self, radars: Dict[int, Radar]):
        """
        初始化雷达网络
        :param radars: 雷达字典 {radar_id: Radar对象}
        """
        self.radars = radars

    def find_covering_radars(self, target_position: np.ndarray) -> list:
        """
        查找所有能覆盖目标的雷达（按空闲通道数排序）
        :param target_position: 目标位置 (x, y, z)
        :return: 覆盖目标的雷达列表（优先返回空闲通道多的雷达）
        """
        covering_radars = [radar for radar in self.radars.values()
                           if radar.is_target_in_range(target_position)]
        # 仅在必要时排序，避免重复排序
        covering_radars.sort(
            key=lambda r: sum(1 for c in r.channels.values() if c is None),
            reverse=True
        )
        logging.debug(f"找到 {len(covering_radars)} 部雷达覆盖目标位置 {target_position}")
        return covering_radars

    def allocate_radar_for_target(self, target_id: int, target_position: np.ndarray) -> Optional[tuple]:
        """
        为目标分配雷达（优先选择空闲通道多的雷达）
        :param target_id: 目标 ID
        :param target_position: 目标位置 (x, y, z)
        :return: (雷达ID, 通道号) 或 None
        """
        covering_radars = self.find_covering_radars(target_position)
        for radar in covering_radars:
            channel_id = radar.allocate_channel(target_id)
            if channel_id is not None:
                return radar.radar_id, channel_id
        logging.warning(f"目标 {target_id} 在位置 {target_position} 无可用雷达通道")
        return None

    def release_radar_channel(self, radar_id: int, channel_id: int):
        """ 释放指定雷达的通道 """
        if radar_id in self.radars:  # 确保雷达存在
            self.radars[radar_id].release_channel(channel_id)
        else:
            logging.error(f"尝试释放不存在的雷达 {radar_id} 的通道 {channel_id}")

    def is_radar_available(self, radar_id: int) -> bool:
        """ 检查指定雷达是否有空闲通道 """
        return radar_id in self.radars and self.radars[radar_id].is_available()
