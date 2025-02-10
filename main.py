import os
import csv
import yaml
import numpy as np
from datetime import datetime
from scipy.sparse import csr_matrix
from typing import Dict, List
from core.models.radar_network import Radar, RadarNetwork
from core.utils.metrics import TrackingMetrics
from visualization.plotter import ResultPlotter
from core.algorithms.bfsa_rho import BFSARHO
from core.algorithms.rule_based import RuleBasedScheduler


# 读取 default.yaml 配置
def load_config(yaml_file: str) -> Dict:
    """ 从 default.yaml 加载实验配置 """
    config_path = os.path.join("data/configs", yaml_file)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# 读取 CSV 文件
def load_csv(file_path: str) -> List[Dict]:
    """ 读取 CSV 文件并转换为字典列表 """
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


# 解析雷达数据
def process_radar_data(radar_data: List[Dict]) -> RadarNetwork:
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
        # 生成模拟目标数据（这里应替换为实际目标提取逻辑）
        current_targets = [{"id": i, "priority": np.random.randint(1, 4), "velocity": np.random.rand(3) * 10} for i in
                           range(num_targets)]
        current_positions = [np.random.rand(3) * 1000 for _ in range(num_targets)]

        # 运行 BFSA-RHO 和 Rule-Based
        bfsa_assignment = bfsa_rho.solve(current_targets, current_positions)
        rule_assignment = rule_based.solve(current_targets, current_positions)

        # 存储当前时间步的分配结果
        bfsa_assignments.append(bfsa_assignment)
        rule_assignments.append(rule_assignment)

    # 7️⃣ 计算评估指标（✅ 修正 `bfsa_assignments` 为空的问题）
    bfsa_report = TrackingMetrics.generate_report(bfsa_assignments[-1], radar_network, current_targets, ΔT)
    rule_based_report = TrackingMetrics.generate_report(rule_assignments[-1], radar_network, current_targets, ΔT)

    # 8️⃣ 可视化并保存图像
    ResultPlotter.plot_weighted_time_comparison(bfsa_report, rule_based_report,
                                                os.path.join(result_dir, "weighted_time.png"))
    ResultPlotter.plot_radar_utilization_heatmap(bfsa_report, rule_based_report, radar_network,
                                                 os.path.join(result_dir, "radar_utilization.png"))
    ResultPlotter.plot_switch_distribution(bfsa_report, rule_based_report,
                                           os.path.join(result_dir, "switch_distribution.png"))
    ResultPlotter.plot_delay_cdf(bfsa_report, rule_based_report, os.path.join(result_dir, "delay_cdf.png"))

    print(f"实验结果已保存至 {result_dir}")


if __name__ == "__main__":
    run_experiment()
