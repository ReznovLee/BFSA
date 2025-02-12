import os
import csv
import yaml
import numpy as np
from datetime import datetime

from numpy import ndarray
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Union
from core.models.radar_network import Radar, RadarNetwork
from core.utils.metrics import TrackingMetrics
from visualization.plotter import ResultPlotter
from core.algorithms.bfsa_rho import BFSARHO
from core.algorithms.rule_based import RuleBasedScheduler
import logging


# 读取 default.yaml 配置
def load_config(yaml_file: str) -> Dict:
    """ 从 default.yaml 加载实验配置 """
    config_path = os.path.join("data/configs", yaml_file)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# 读取 CSV 文件
def load_csv(file_path: str) -> List[Dict[str, str]]:
    """ 读取 CSV 文件并转换为字典列表 """
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


# 解析雷达数据
def process_radar_data(radar_data: List[Dict[str, str]]) -> RadarNetwork:
    """ 解析雷达数据并创建雷达网络对象 """
    radars = {}
    for row in radar_data:
        radar_id = int(row["id"])
        radars[radar_id] = Radar(
            radar_id=radar_id,
            position=np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
            detection_range=float(row["radius"]),
            num_channels=int(row["number_channel"])
        )
    return RadarNetwork(radars)

"""
# 解析目标数据
def process_target_data(target_data: List[Dict[str, str]], radar_network: RadarNetwork) -> tuple[
    list[dict[str, Union[int, ndarray]]], list[ndarray]]:
    # 解析目标数据并返回目标字典列表 
    targets = []
    target_positions = []
    for row in target_data:
        targets.append({
            "id": int(row["id"]),
            "priority": int(row["priority"]),
            "velocity": np.array([float(row["vx"]), float(row["vy"]), float(row["vz"])]),
        })
        target_positions.append(np.array([float(row["x"]), float(row["y"]), float(row["z"])]))
    return targets, target_positions
"""

def process_target_data(target_data: List[Dict[str, str]], radar_network: RadarNetwork) -> tuple[
    list[dict[str, Union[int, ndarray]]], list[ndarray]]:
    """ 解析目标数据并返回目标字典列表 """
    targets = []
    target_positions = []
    for row in target_data:
        target_id = int(row["id"])
        target_pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])

        targets.append({
            "id": target_id,
            "priority": int(row["priority"]),
            "velocity": np.array([float(row["vx"]), float(row["vy"]), float(row["vz"])]),
        })
        target_positions.append(target_pos)

        # Debug: 检查目标是否被任何雷达覆盖
        covering_radars = radar_network.find_covering_radars(target_pos)
        if covering_radars:
            logging.debug(f"目标 {target_id} 被以下雷达覆盖: {[r.radar_id for r in covering_radars]}")
        else:
            logging.warning(f"目标 {target_id} 在位置 {target_pos} 未被任何雷达覆盖")

    return targets, target_positions


# 获取最新 scenario 目录
def get_latest_scenario_folder() -> str:
    """ 查找最新生成的 scenario 目录 """
    scenario_base_path = "data/generators"
    scenario_folders = sorted(
        [d for d in os.listdir(scenario_base_path) if d.startswith("scenario-")],
        reverse=True
    )
    if scenario_folders:
        return os.path.join(scenario_base_path, scenario_folders[0])
    raise FileNotFoundError("No scenario folder found in 'data/generators'.")


# 存储分配结果
def save_assignment_results(assignments: List[csr_matrix], result_dir: str, algorithm_name: str):
    """ 将调度分配结果保存为 CSV """
    assignment_file = os.path.join(result_dir, f"{algorithm_name}_assignments.csv")
    with open(assignment_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time Step", "Target ID", "Radar ID"])
        for t, assignment in enumerate(assignments):
            rows, cols = assignment.nonzero()
            for r, c in zip(rows, cols):
                writer.writerow([t, r, c])


# 运行实验
def run_experiment():
    """ 主实验入口，执行 BFSA-RHO 和 Rule-Based 算法并进行可视化 """

    # 1️⃣ 读取配置
    config = load_config("default.yaml")
    Δt = config["simulation"]["time_step"]
    ΔT = config["simulation"]["algorithm_step"]
    num_radars = config["num_radars"]
    num_targets = config["num_targets"]

    # 2️⃣ 获取最新 scenario 目录
    latest_scenario = get_latest_scenario_folder()
    target_file = os.path.join(latest_scenario, f"{num_targets}-targets.csv")
    radar_file = os.path.join(latest_scenario, f"{num_radars}-radar.csv")

    # 3️⃣ 读取数据
    radar_network = process_radar_data(load_csv(radar_file))
    targets, target_positions = process_target_data(load_csv(target_file), radar_network)

    # 4️⃣ 初始化算法调度器
    bfsa_rho = BFSARHO(radar_network)
    rule_based = RuleBasedScheduler(radar_network)

    # 5️⃣ 创建结果存储路径
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_dir = os.path.join("results", f"{num_radars}R{num_targets}T-result {timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # 6️⃣ 逐步运行算法（每 ΔT 执行一次）
    bfsa_assignments, rule_assignments = [], []
    time_steps = list(range(0, int(config["simulation"]["total_time"]), int(ΔT)))

    for t in time_steps:
        # 运行 BFSA-RHO 和 Rule-Based
        bfsa_assignment = bfsa_rho.solve(targets, target_positions)
        rule_assignment = rule_based.solve(targets, target_positions)

        # 存储当前时间步的分配结果
        bfsa_assignments.append(bfsa_assignment)
        rule_assignments.append(rule_assignment)

        # 调试输出每个时间步的分配矩阵，检查算法的输出是否正常
        print(
            f"Time step {t} BFSA-RHO Assignment: {bfsa_assignment.shape}, Rule-Based Assignment: {rule_assignment.shape}")

    # 7️⃣ 存储调度结果
    save_assignment_results(bfsa_assignments, result_dir, "BFSA-RHO")
    save_assignment_results(rule_assignments, result_dir, "Rule-Based")

    # 8️⃣ 计算评估指标
    bfsa_report = TrackingMetrics.generate_report(bfsa_assignments[-1], radar_network, targets, ΔT)
    rule_based_report = TrackingMetrics.generate_report(rule_assignments[-1], radar_network, targets, ΔT)

    # 9️⃣ 可视化并保存图像
    ResultPlotter.plot_weighted_time_comparison(bfsa_report, rule_based_report,
                                                os.path.join(result_dir, "weighted_time.png"))
    ResultPlotter.plot_radar_utilization_heatmap(bfsa_report, rule_based_report, radar_network,
                                                 os.path.join(result_dir, "radar_utilization.png"))
    ResultPlotter.plot_switch_distribution(bfsa_report, rule_based_report,
                                           os.path.join(result_dir, "switch_distribution.png"))
    ResultPlotter.plot_delay_cdf(bfsa_report, rule_based_report, os.path.join(result_dir, "delay_cdf.png"))

    # 10️⃣ 绘制甘特图
    ResultPlotter.plot_gantt_chart(bfsa_assignments, time_steps, mode="target",
                                   save_path=os.path.join(result_dir, "target_gantt_chart.png"))
    ResultPlotter.plot_gantt_chart(bfsa_assignments, time_steps, mode="radar",
                                   save_path=os.path.join(result_dir, "radar_gantt_chart.png"))

    print(f"实验结果已保存至 {result_dir}")


if __name__ == "__main__":
    run_experiment()
