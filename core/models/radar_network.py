from typing import Dict
import numpy as np


class Radar:
    """ 单部雷达类 """

    def __init__(self, radar_id, position, num_channels, detection_range):
        """
        初始化雷达
        :param radar_id: 雷达唯一ID
        :param position: 雷达位置 (x, y, z)
        :param num_channels: 雷达的通道数
        :param detection_range: 雷达的探测范围 (m)
        """
        self.radar_id = radar_id
        self.position = np.array(position, dtype=np.float64)
        self.num_channels = num_channels
        self.detection_range = detection_range  #
        self.channels = {i: None for i in range(num_channels)}  # 通道状态

    def is_available(self):
        """ 检查是否有空闲通道 """
        return any(channel is None for channel in self.channels.values())

    def allocate_channel(self, target_id):
        """ 为目标分配空闲通道 """
        for channel_id in range(self.num_channels):
            if self.channels[channel_id] is None:
                self.channels[channel_id] = target_id
                return channel_id
        return None  # 无可用通道

    def release_channel(self, channel_id):
        """ 释放通道 """
        self.channels[channel_id] = None

    def is_target_in_range(self, target_position):
        """
        判断目标是否在雷达的探测范围内
        :param target_position: 目标位置 (x, y, z)
        :return: 是否在探测范围内
        """
        distance = np.linalg.norm(self.position - np.array(target_position))
        return distance <= self.detection_range


class RadarNetwork:
    """ 雷达网络类 """

    def __init__(self, radars: Dict[int, Radar]):
        """
        初始化雷达网络
        :param radars: 雷达字典 {radar_id: Radar对象}
        """
        self.radars = radars

    def find_covering_radars(self, target_position):
        """
        查找所有能覆盖目标的雷达
        :param target_position: 目标位置 (x, y, z)
        :return: 覆盖目标的雷达列表
        """
        return [radar for radar in self.radars.values() if radar.is_target_in_range(target_position)]

    def allocate_radar_for_target(self, target_id, target_position):
        """
        为目标分配雷达
        :param target_id: 目标 ID
        :param target_position: 目标位置 (x, y, z)
        :return: (雷达ID, 通道号) 或 None
        """
        for radar in self.find_covering_radars(target_position):
            channel_id = radar.allocate_channel(target_id)
            if channel_id is not None:
                return radar.radar_id, channel_id
        return None  # 无可用雷达

    def release_radar_channel(self, radar_id, channel_id):
        """ 释放指定雷达的通道 """
        if radar_id in self.radars:
            self.radars[radar_id].release_channel(channel_id)

    def is_radar_available(self, radar_id: int) -> bool:
        """ 检查指定雷达是否有空闲通道 """
        return radar_id in self.radars and self.radars[radar_id].is_available()
