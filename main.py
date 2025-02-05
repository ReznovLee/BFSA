from typing import Dict
import yaml
import numpy as np
from scipy.sparse import vstack
from core.models.radar_network import RadarNetwork
from data.generators.scenario_generator import ScenarioGenerator
from core.algorithms.bfsa_rho import BFSARHO
from core.algorithms.rule_based import RuleBasedScheduler
from core.utils.metrics import TrackingMetrics
from visualization.plotter import ResultPlotter


def load_config(config_path: str = "data/configs/default.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    delta_t = config['simulation']["delta_t"]
    delta_T = config['simulation']["delta_T"]
    total_time = config['simulation']["total_time"]  # 总仿真时间

    # 生成场景
    generator = ScenarioGenerator()
    targets, radars = generator.generate_scenario(
        num_targets=5,
        num_radars=3,
        total_time=total_time,
        save_csv=True,
        output_dir="data/scenario_1"
    )
    radar_network = RadarNetwork(radars)

    # 初始化算法
    bfsa = BFSARHO(radar_network, alpha=0.9, gamma=0.8)
    rule_based = RuleBasedScheduler(radar_network)

    # 仿真循环
    current_time = 0.0
    bfsa_assignments = []
    rule_assignments = []
    schedule_steps = []

    while current_time <= total_time:
        # 到达调度时刻时执行分配
        if np.isclose(current_time % delta_T, 0):
            # 获取当前目标位置（最新状态）
            # target_positions = [t["model"].trajectory[-1]["state"][:3] for t in targets]
            target_positions = [t["trajectory"][-1]["position"] for t in targets]

            # BFSA-RHO分配
            bfsa_assignment = bfsa.solve(targets, target_positions)
            bfsa_assignments.append(bfsa_assignment)

            # 规则算法分配
            rule_assignment = rule_based.solve(targets, target_positions)
            rule_assignments.append(rule_assignment)

            schedule_steps.append(current_time)

        current_time += delta_t

    # 性能评估
    bfsa_report = TrackingMetrics.generate_report(
        vstack(bfsa_assignments), radar_network, targets, time_step=delta_T
    )
    rule_report = TrackingMetrics.generate_report(
        vstack(rule_assignments), radar_network, targets, time_step=delta_T
    )

    # 可视化对比
    ResultPlotter.plot_weighted_time_comparison(bfsa_report, rule_report)
    ResultPlotter.plot_radar_utilization_heatmap(bfsa_report, rule_report, radar_network)


if __name__ == "__main__":
    main()
