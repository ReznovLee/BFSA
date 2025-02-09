import os
import csv
import yaml
import numpy as np
from datetime import datetime
from scipy.sparse import csr_matrix
from typing import Dict, List
from core.models.radar_network import RadarNetwork
from core.utils.metrics import TrackingMetrics
from visualization.plotter import ResultPlotter
from core.algorithms.bfsa_rho import BFSARHO
from core.algorithms.rule_based import RuleBasedScheduler

# 读取 default.yaml 配置
def load_config(yaml_file: str) -> Dict:
    """ 从 default.yaml 加载实验配置 """
    config_path = os.path.join("../configs", yaml_file)  # 统一管理配置路径
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# 读取 CSV 文件（雷达 or 目标）
def load_csv(file_path: str) -> List[Dict]:
    """ 读取 CSV 文件并转换为字典列表 """
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

# 解析目标数据
def process_target_data(targets: List[Dict]) -> Dict[int, List[Dict]]:
    """ 将目标数据按照目标 ID 进行分组（方便按 ΔT 运行算法）"""
    target_dict = {}
    for row in targets:
        tid = int(row["id"])
        if tid not in target_dict:
            target_dict[tid] = []
        target_dict[tid].append({
            "timestamp": float(row["timestep"]),
            "position": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
            "velocity": np.array([float(row["vx"]), float(row["vy"]), float(row["vz"])]),
            "priority": int(row["priority"])
        })
    return target_dict

# 解析雷达数据
def process_radar_data(radar_data: List[Dict]) -> RadarNetwork:
    """ 解析雷达数据并创建雷达网络对象 """
    radar_network = RadarNetwork()
    for row in radar_data:
        radar_network.add_radar(
            radar_id=int(row["id"]),
            position=np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
            radius=float(row["radius"]),
            num_channels=int(row["number_channel"])
        )
    return radar_network

# 运行实验
def run_experiment():
    """ 主实验入口，执行 BFSA-RHO 和 Rule-Based 算法并进行可视化 """
    
    # 1️⃣ 读取配置文件
    config = load_config("default.yaml")
    Δt = config["simulation"]["time_step"]  # 目标数据的时间步长
    ΔT = config["simulation"]["algorithm_step"]  # 运行算法的时间间隔（ΔT 是 Δt 的整数倍）
    num_radars = config["num_radars"]
    num_targets = config["num_targets"]

    # 2️⃣ 读取目标和雷达数据
    target_file = f"../data/{num_targets}-targets.csv"
    radar_file = f"../data/{num_radars}-radar.csv"
    
    target_data = process_target_data(load_csv(target_file))
    radar_network = process_radar_data(load_csv(radar_file))

    # 3️⃣ 初始化算法调度器
    bfsa_rho = BFSARHO(radar_network)
    rule_based = RuleBasedScheduler(radar_network)

    # 4️⃣ 创建结果存储路径
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_dir = f"../results/{num_radars}R{num_targets}T-result {timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # 5️⃣ 逐步运行算法（每 ΔT 执行一次）
    time_steps = sorted(set(float(row["timestep"]) for row in target_data[1]))  # 获取时间步
    time_steps = time_steps[::int(ΔT / Δt)]  # 仅在 ΔT 运行
    
    bfsa_assignments, rule_assignments = [], []
    
    for t in time_steps:
        # 提取当前时间步的目标状态
        current_targets = []
        current_positions = []
        
        for tid, traj in target_data.items():
            for state in traj:
                if state["timestamp"] == t:
                    current_targets.append({
                        "id": tid,
                        "priority": state["priority"]
                    })
                    current_positions.append(state["position"])
                    break
        
        # 运行 BFSA-RHO 和 Rule-Based
        bfsa_assignment = bfsa_rho.solve(current_targets, current_positions)
        rule_assignment = rule_based.solve(current_targets, current_positions)
        
        # 存储当前时间步的分配结果
        bfsa_assignments.append(bfsa_assignment)
        rule_assignments.append(rule_assignment)

    # 6️⃣ 计算评估指标
    bfsa_report = TrackingMetrics.generate_report(bfsa_assignments[-1], radar_network, current_targets, ΔT)
    rule_based_report = TrackingMetrics.generate_report(rule_assignments[-1], radar_network, current_targets, ΔT)

    # 7️⃣ 可视化并保存图像
    ResultPlotter.plot_weighted_time_comparison(bfsa_report, rule_based_report, os.path.join(result_dir, "weighted_time.png"))
    ResultPlotter.plot_radar_utilization_heatmap(bfsa_report, rule_based_report, radar_network, os.path.join(result_dir, "radar_utilization.png"))
    ResultPlotter.plot_coverage_timeline(bfsa_assignments, rule_assignments, time_steps, os.path.join(result_dir, "coverage_timeline.png"))
    ResultPlotter.plot_switch_distribution(bfsa_report, rule_based_report, os.path.join(result_dir, "switch_distribution.png"))
    ResultPlotter.plot_target_assignment_timeline(bfsa_assignments, time_steps, os.path.join(result_dir, "assignment_timeline.png"))
    ResultPlotter.plot_delay_cdf(bfsa_report, rule_based_report, os.path.join(result_dir, "delay_cdf.png"))

    print(f"实验结果已保存至 {result_dir}")

if __name__ == "__main__":
    run_experiment()
