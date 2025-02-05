import numpy as np
import yaml
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class ScenarioGenerator:
    """
    场景生成器：根据参数动态生成目标轨迹和雷达网络，支持覆盖配置中的默认值。
    """

    def __init__(self, config_path: str = "data/configs/default.yaml"):
        self.config = self._load_config(config_path)
        self.rng = np.random.default_rng(self.config["simulation"]["seed"])
        self.airspace = self.config["targets"]["airspace_bounds"]
        self.target_params = self.config["targets"]
        self.radar_params = self.config["radars"]

    def generate_scenario(
            self,
            save_csv: bool = True,
            output_dir: str = "data/generated",
            num_targets: Optional[int] = None,
            num_radars: Optional[int] = None,
            total_time: Optional[int] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        生成目标轨迹和雷达网络，允许覆盖默认配置参数。
        :param num_targets: 目标数量（覆盖配置文件）
        :param num_radars: 雷达数量（覆盖配置文件）
        :param total_time: 总时间步数（覆盖配置文件）
        """
        # 动态覆盖配置参数（如果未传入则使用配置文件的值）
        effective_num_targets = num_targets if num_targets is not None else self.target_params["num_targets"]
        effective_num_radars = num_radars if num_radars is not None else self.radar_params["num_radars"]
        effective_total_time = total_time if total_time is not None else self.config["simulation"]["total_time"]

        # 生成目标轨迹
        targets = self._generate_targets(effective_num_targets, effective_total_time)

        # 生成雷达网络（确保覆盖目标轨迹）
        radars = self._generate_radars(effective_num_radars, targets)

        # 保存数据（可选）
        if save_csv:
            self._save_to_csv(targets, radars, output_dir)
        return targets, radars

    def _generate_targets(self, num_targets: int, total_time: int) -> List[Dict]:
        """生成目标列表，按类型分布均匀分配空域"""
        targets = []
        num_per_type = self._calculate_target_numbers(num_targets)

        for target_type, count in num_per_type.items():
            for _ in range(count):
                target_id = len(targets)
                position = self._generate_target_position(target_type)
                velocity = self._generate_target_velocity(target_type)
                trajectory = self._simulate_trajectory(target_type, position, velocity, total_time)
                targets.append({
                    "id": target_id,
                    "type": target_type,
                    "priority": self._get_priority(target_type),
                    "trajectory": trajectory
                })
        return targets

    def _calculate_target_numbers(self, num_targets: int) -> Dict[str, int]:
        """计算各类型目标数量（按比例分配）"""
        ratios = self.target_params["type_distribution"]
        counts = {
            "ballistic": int(num_targets * ratios["ballistic"]),
            "cruise": int(num_targets * ratios["cruise"]),
            "aircraft": int(num_targets * ratios["aircraft"])
        }
        counts["ballistic"] += num_targets - sum(counts.values())
        return counts

    def _simulate_trajectory(
            self,
            target_type: str,
            initial_pos: np.ndarray,
            initial_vel: np.ndarray,
            total_time: int
    ) -> List[Dict]:
        """模拟目标轨迹，使用传入的 total_time 覆盖配置"""
        dt = self.config["simulation"]["delta_t"]
        trajectory = []
        pos = initial_pos.copy()
        vel = initial_vel.copy()

        for t in range(total_time):
            # 更新状态（根据目标类型）
            if target_type == "ballistic":
                pos[2] += vel[2] * dt + 0.5 * 9.81 * dt ** 2
                vel[2] += 9.81 * dt
            elif target_type == "cruise":
                if t < 0.7 * total_time:
                    noise = self.rng.normal(0, 0.1, 3)
                    vel += noise * dt
                else:
                    pos[2] += vel[2] * dt + 0.5 * 9.81 * dt ** 2
                    vel[2] += 9.81 * dt
            elif target_type == "aircraft":
                noise = self.rng.normal(0, 0.2, 3)
                vel += noise * dt
                pos[2] = np.clip(pos[2] + vel[2] * dt, 500, self.airspace["z"]["aircraft"][1])

            pos[:2] += vel[:2] * dt
            trajectory.append({
                "time": t * self.config["simulation"]["delta_t"],
                "position": pos.copy().tolist(),
                "velocity": vel.copy().tolist()
            })
        return trajectory

    def _generate_radars(self, num_radars: int, targets: List[Dict]) -> List[Dict]:
        """生成雷达网络，数量由参数动态指定"""
        radars = []
        for radar_id in range(num_radars):
            position = self._generate_radar_position(targets)
            radius = self.rng.integers(*self.radar_params["radius_range"])
            channels = self.rng.integers(*self.radar_params["channels_range"])
            radars.append({
                "id": radar_id,
                "position": position,
                "radius": radius,
                "channels": channels
            })
        return radars

    # 其余辅助方法（_generate_target_position, _generate_radar_position, _save_to_csv等）保持不变

    def _generate_target_position(self, target_type: str) -> np.ndarray:
        """生成目标的初始位置（均匀分布在指定空域层）"""
        z_min, z_max = self.airspace["z"][target_type]
        x = self.rng.uniform(*self.airspace["x"])
        y = self.rng.uniform(*self.airspace["y"])
        z = self.rng.uniform(z_min, z_max)
        return np.array([x, y, z])

    def _generate_target_velocity(self, target_type: str) -> np.ndarray:
        """生成目标的初始速度（根据类型范围随机）"""
        vx_range = self.target_params["velocity_ranges"][target_type]["vx"]
        vy_range = self.target_params["velocity_ranges"][target_type]["vy"]
        vz_range = self.target_params["velocity_ranges"][target_type]["vz"]
        vx = self.rng.uniform(*vx_range)
        vy = self.rng.uniform(*vy_range)
        vz = self.rng.uniform(*vz_range)
        return np.array([vx, vy, vz])

    def _generate_radar_position(self, targets: List[Dict]) -> List[float]:
        """生成雷达位置，确保覆盖至少一个目标点（在半球范围内）"""
        for _ in range(100):  # 最大尝试次数
            x = self.rng.uniform(*self.airspace["x"])
            y = self.rng.uniform(*self.airspace["y"])
            z = self.radar_params["position_z"]
            # 检查是否覆盖至少一个目标点
            for target in targets:
                for point in target["trajectory"]:
                    px, py, pz = point["position"]
                    distance = np.sqrt((px - x)**2 + (py - y)**2 + (pz - z)**2)
                    if distance <= self.radar_params["radius_range"][1] and pz >= z:
                        return [x, y, z]
        # 若无法满足，返回随机位置
        return [x, y, z]

    def _get_priority(self, target_type: str) -> int:
        """获取目标威胁优先级"""
        return 1 if target_type == "ballistic" else 2 if target_type == "cruise" else 3

    def _save_to_csv(self, targets: List[Dict], radars: List[Dict], output_dir: str):
        """保存为CSV文件"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # 保存雷达配置
        radar_path = os.path.join(output_dir, "radars.csv")
        with open(radar_path, "w") as f:
            f.write("id,x,y,z,radius,channels\n")
            for r in radars:
                f.write(f"{r['id']},{r['position'][0]},{r['position'][1]},{r['position'][2]},{r['radius']},{r['channels']}\n")
        # 保存目标轨迹
        for target in targets:
            target_path = os.path.join(output_dir, f"target_{target['id']}.csv")
            with open(target_path, "w") as f:
                f.write("time,x,y,z,vx,vy,vz,priority,type\n")
                for point in target["trajectory"]:
                    f.write(f"{point['time']},{point['position'][0]},{point['position'][1]},{point['position'][2]},"
                            f"{point['velocity'][0]},{point['velocity'][1]},{point['velocity'][2]},"
                            f"{target['priority']},{target['type']}\n")

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config