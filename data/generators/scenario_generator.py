import os
import csv
import yaml
import random
import numpy as np
from datetime import datetime
from core.models.radar_network import Radar
from core.models.target_model import BallisticMissile, CruiseMissile, FighterJet


# 读取 default.yaml 获取配置
def load_config(filename):
    """ 从 ../configs/ 目录加载 default.yaml """
    config_path = '../configs/' + filename  # 通过相对路径访问
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# 生成随机雷达数据（固定 z = 0）
def generate_random_radars(num_radars):
    """ 生成随机雷达信息（z 坐标固定为 0） """
    radars = []
    for i in range(num_radars):
        position = [random.uniform(0, 10000), random.uniform(0, 10000), 0]  # z 坐标固定为 0
        detection_range = random.uniform(5000, 15000)  # 随机探测范围
        num_channels = random.randint(2, 6)  # 随机通道数

        radar = Radar(i, position, num_channels, detection_range)
        radars.append(radar)

    return radars


# 将雷达信息存入 CSV
def save_radars_to_csv(radars, folder, radar_filename):
    """ 存储雷达数据到 CSV """
    os.makedirs(folder, exist_ok=True)
    radar_file_path = os.path.join(folder, radar_filename)

    with open(radar_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "x", "y", "z", "radius", "number_channel"])

        for radar in radars:
            writer.writerow([radar.radar_id, *radar.position, radar.detection_range, radar.num_channels])


# 计算目标数量
def compute_target_counts(num_targets, target_ratio):
    """ 根据目标比例计算每类目标的数量 """
    ballistic_count = int(num_targets * target_ratio["ballistic_missile"])
    cruise_count = int(num_targets * target_ratio["cruise_missile"])
    fighter_count = num_targets - ballistic_count - cruise_count  # 确保总数匹配
    return {
        "ballistic_missile": ballistic_count,
        "cruise_missile": cruise_count,
        "fighter_jet": fighter_count
    }


# 生成随机目标数据
def generate_random_targets(target_counts):
    """ 生成目标信息，并分配 ID """
    targets = []
    target_id = 1  # 目标 ID 从 1 开始
    for target_type, count in target_counts.items():
        for _ in range(count):
            position = [random.uniform(0, 10000) for _ in range(3)]  # 随机位置
            velocity_mach = [random.uniform(0.5, 3.0) for _ in range(3)]  # 随机速度 (Mach)

            if target_type == "ballistic_missile":
                target = BallisticMissile(target_id, position, velocity_mach)
            elif target_type == "cruise_missile":
                target = CruiseMissile(target_id, position, velocity_mach)
            elif target_type == "fighter_jet":
                target = FighterJet(target_id, position, velocity_mach)
            else:
                continue  # 忽略无效类型

            targets.append(target)
            target_id += 1  # 目标 ID 递增

    return targets


# 生成目标轨迹并存入 CSV
def save_targets_to_csv(targets, folder, targets_filename, total_time=100, time_step=0.1):
    """ 生成目标轨迹并存入 CSV，确保相同 ID 先排列 """
    os.makedirs(folder, exist_ok=True)
    target_file_path = os.path.join(folder, targets_filename)

    with open(target_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "timestep", "x", "y", "z", "vx", "vy", "vz", "target_type", "priority"])

        for target in targets:  # 先遍历目标 ID，确保相同 ID 的数据连在一起
            for t in np.arange(0, total_time, time_step):
                target.update_position(time_step)
                writer.writerow(target.get_state(t))


# 主程序：根据 default.yaml 生成场景
def generate_scenario():
    """ 读取 default.yaml 并生成雷达和目标数据 """
    # 读取配置
    config = load_config("default.yaml")

    num_radars = config["num_radars"]
    num_targets = config["num_targets"]
    target_ratio = config["target_ratio"]

    # 计算各类目标数量
    target_counts = compute_target_counts(num_targets, target_ratio)

    # 获取当前日期，作为文件夹名称
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_folder = f"scenario-{current_date}"

    # 生成雷达数据
    radars = generate_random_radars(num_radars)
    radar_filename = config["output"]["radar_filename_template"].format(num_radars=num_radars)
    save_radars_to_csv(radars, output_folder, radar_filename)

    # 生成目标数据
    targets = generate_random_targets(target_counts)
    targets_filename = config["output"]["target_filename_template"].format(num_targets=num_targets)
    save_targets_to_csv(targets, output_folder, targets_filename)


if __name__ == "__main__":
    generate_scenario()
