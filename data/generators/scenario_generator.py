import os
import csv
import yaml
import random
import numpy as np
from datetime import datetime
from core.models.radar_network import Radar
from target_model import BallisticMissile, CruiseMissile, FighterJet

# 读取 default.yaml 获取配置
def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# 生成随机雷达数据
def generate_random_radars(num_radars):
    radars = []
    for i in range(num_radars):
        position = [random.uniform(0, 10000) for _ in range(3)]  # 随机坐标
        detection_range = random.uniform(5000, 15000)  # 随机探测范围
        num_channels = random.randint(2, 6)  # 随机通道数

        radar = Radar(i, position, num_channels, detection_range)
        radars.append(radar)
    
    return radars

# 将雷达信息存入 CSV
def save_radars_to_csv(radars, folder, radar_filename):
    os.makedirs(folder, exist_ok=True)
    radar_file_path = os.path.join(folder, radar_filename)
    
    with open(radar_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "x", "y", "z", "radius", "number_channel"])
        
        for radar in radars:
            writer.writerow([radar.radar_id, *radar.position, radar.detection_range, radar.num_channels])

# 生成随机目标数据
def generate_random_targets(num_targets, target_ratio):
    targets = []
    target_counts = {
        "ballistic_missile": int(num_targets * target_ratio["ballistic_missile"]),
        "cruise_missile": int(num_targets * target_ratio["cruise_missile"]),
        "fighter_jet": int(num_targets * target_ratio["fighter_jet"]),
    }

    # 根据目标比例生成目标
    for target_type, count in target_counts.items():
        for i in range(count):
            position = [random.uniform(0, 10000) for _ in range(3)]  # 随机位置
            velocity_mach = [random.uniform(0.5, 3.0) for _ in range(3)]  # 随机速度 (Mach)

            if target_type == "ballistic_missile":
                target = BallisticMissile(i, position, velocity_mach)
            elif target_type == "cruise_missile":
                target = CruiseMissile(i, position, velocity_mach)
            elif target_type == "fighter_jet":
                target = FighterJet(i, position, velocity_mach)
            else:
                continue  # 忽略无效类型

            targets.append(target)
    
    return targets

# 生成目标轨迹并存入 CSV
def save_targets_to_csv(targets, folder, targets_filename, total_time=100, time_step=1):
    os.makedirs(folder, exist_ok=True)
    target_file_path = os.path.join(folder, targets_filename)
    
    with open(target_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "timestep", "x", "y", "z", "vx", "vy", "vz", "target_type", "priority"])

        for t in range(0, total_time, time_step):
            for target in targets:
                target.update_position(time_step)
                writer.writerow(target.get_state(t))

# 主程序：根据 default.yaml 生成场景
def generate_scenario():
    # 读取配置
    config = load_config("default.yaml")

    num_radars = config["num_radars"]
    num_targets = config["num_targets"]
    target_ratio = config["target_ratio"]

    # 获取当前日期，作为文件夹名称
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_folder = f"scenario-{current_date}"

    # 生成雷达数据
    radars = generate_random_radars(num_radars)
    radar_filename = config["output"]["radar_filename_template"].format(num_radars=num_radars)
    save_radars_to_csv(radars, output_folder, radar_filename)

    # 生成目标数据
    targets = generate_random_targets(num_targets, target_ratio)
    targets_filename = config["output"]["target_filename_template"].format(num_targets=num_targets)
    save_targets_to_csv(targets, output_folder, targets_filename)

if __name__ == "__main__":
    generate_scenario()
