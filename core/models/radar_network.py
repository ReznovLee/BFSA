import numpy as np
from typing import List, Dict, Optional


class Radar:
    """
    单部雷达模型，管理雷达状态和通道占用。
    文档定义：雷达为陆基固定雷达，覆盖范围为以雷达位置为中心、垂直向上的半球。
    """

    def __init__(
            self,
            radar_id: int,
            position: np.ndarray,
            radius: float,
            channels: int,
            min_height: float = 0.0
    ):
        """
        :param radar_id: 雷达唯一标识
        :param position: 雷达坐标 [x, y, z]（单位：米）
        :param radius: 探测半径（单位：米）
        :param channels: 最大跟踪通道数
        :param min_height: 最低探测高度（默认0，即陆基雷达）
        """
        self.id = radar_id
        self.position = position.astype(float)
        self.radius = radius
        self.channels = channels
        self.min_height = min_height
        self.occupied_channels = 0  # 当前占用的通道数

    def is_available(self) -> bool:
        """检查雷达是否有空闲通道"""
        return self.occupied_channels < self.channels

    def allocate_channel(self) -> bool:
        """分配一个通道，成功返回True，失败返回False"""
        if self.is_available():
            self.occupied_channels += 1
            return True
        return False

    def release_channel(self) -> None:
        """释放一个通道（如果已被占用）"""
        if self.occupied_channels > 0:
            self.occupied_channels -= 1

    def is_target_in_range(self, target_position: np.ndarray) -> bool:
        """
        检查目标是否在雷达覆盖范围内（半球范围）
        文档定义：覆盖范围为以雷达位置为中心、z轴正方向的半球。
        """
        delta = target_position - self.position
        horizontal_distance = np.linalg.norm(delta[:2])  # x-y平面距离
        vertical_distance = delta[2]  # z方向高度差

        # 条件1：水平距离 <= 探测半径
        # 条件2：目标高度 >= 雷达高度（z轴正方向）
        # 条件3：总距离（三维） <= 探测半径（冗余检查）
        return (
                horizontal_distance <= self.radius and
                vertical_distance >= -self.min_height and
                np.linalg.norm(delta) <= self.radius
        )


class RadarNetwork:
    """
    雷达网络模型，管理多部雷达的状态和协同调度。
    文档定义：雷达网络需要保证同一空域中至少有两部雷达可覆盖目标。
    """

    def __init__(self, radars: List[Dict]):
        """
        :param radars: 雷达字典列表，每个字典需包含以下键：
            - 'id': 雷达唯一标识（整数）
            - 'position': 雷达坐标 [x, y, z]
            - 'radius': 侦测半径（米）
            - 'channels': 最大通道数
        """
        self.radars = {r['id']: r for r in radars}  # 按ID存储雷达字典
        self.occupied_channels = {r['id']: 0 for r in radars}  # 各雷达已占用通道数

    def find_covering_radars(self, target_position: np.ndarray) -> List[Radar]:
        """返回所有能覆盖目标的雷达列表（按距离排序）"""
        covering = []
        for radar in self.radars.values():
            if is_target_in_range(radar, target_position):
                covering.append(radar)
        # 按到目标的水平距离排序（优先近端雷达）
        covering.sort(key=lambda r: np.linalg.norm(target_position[:2] - r.position[:2]))
        return covering

    def allocate_radar_for_target(
            self,
            target_position: np.ndarray,
            exclude_radars: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        为目标分配一个雷达（基于就近原则和通道可用性）
        :return: 分配的雷达ID，若无可用雷达则返回None
        """
        covering_radars = self.find_covering_radars(target_position)
        if exclude_radars:
            covering_radars = [r for r in covering_radars if r.id not in exclude_radars]

        for radar in covering_radars:
            if radar.allocate_channel():
                return radar.id
        return None

    def release_radar_channel(self, radar_id: int) -> None:
        """释放指定雷达的一个通道"""
        if radar_id in self.radars:
            self.radars[radar_id].release_channel()

    def reset_channels(self) -> None:
        """重置所有雷达的通道占用（用于仿真重置）"""
        for radar in self.radars.values():
            radar.occupied_channels = 0

    @staticmethod
    def generate_random_radar(
            radar_id: int,
            position_range: tuple = (-10000, 10000),
            radius_range: tuple = (5000, 10000),
            channels_range: tuple = (5, 10)
    ) -> Radar:
        """
        生成随机参数的雷达（用于仿真测试）
        :param position_range: 雷达坐标范围 (x_min, x_max)
        :param radius_range: 探测半径范围 (min_radius, max_radius)
        :param channels_range: 通道数范围 (min_channels, max_channels)
        """
        x = np.random.uniform(position_range[0], position_range[1])
        y = np.random.uniform(position_range[0], position_range[1])
        z = 0  # 陆基雷达
        radius = np.random.uniform(radius_range[0], radius_range[1])
        channels = np.random.randint(channels_range[0], channels_range[1] + 1)
        return Radar(radar_id, np.array([x, y, z]), radius, channels)

    def is_radar_available(self, radar_id: int) -> bool:
        """检查雷达是否有空闲通道"""
        return self.occupied_channels[radar_id] < self.radars[radar_id]['channels']

def is_target_in_range(radar: dict, target_position: np.ndarray) -> bool:
    """
    判断目标是否在雷达的半球范围内。
    :param radar: 雷达字典，必须包含 'position' 和 'radius' 键。
    :param target_position: 目标位置 [x, y, z]。
    :return: True（在范围内）/ False（不在范围内）。
    """
    # 计算目标与雷达的距离
    delta = np.array(target_position) - np.array(radar["position"])
    distance = np.linalg.norm(delta)
    # 检查距离是否小于雷达半径，且目标高度不低于雷达（半球范围）
    return distance <= radar["radius"] and target_position[2] >= radar["position"][2]