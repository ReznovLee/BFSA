import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from scipy.sparse import csr_matrix
from core.models.radar_network import RadarNetwork
import os
import logging


class ResultPlotter:
    """
    结果可视化工具，支持 BFSA-RHO 与 Rule-Based 算法对比：
    1. 加权总跟踪时间对比（柱状图）
    2. 雷达通道利用率（热力图）
    3. 目标覆盖比例随时间变化（折线图）
    4. 跟踪切换次数分布（箱线图 + 均值线）
    5. 目标-雷达分配随时间变化（动态图）
    6. 首次跟踪延迟分布（CDF）
    7. 调度甘特图（目标 vs 时间）
    8. 调度甘特图（雷达 vs 时间）
    """

    @staticmethod
    def plot_weighted_time_comparison(bfsa_report: Dict, rule_based_report: Dict, save_path: str = None) -> None:
        """绘制加权总跟踪时间对比（柱状图）"""
        labels = ['BFSA-RHO', 'Rule-Based']
        values = np.array([bfsa_report['weighted_total_time'], rule_based_report['weighted_total_time']])

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=['#2ecc71', '#e74c3c'])
        plt.ylabel('Weighted Total Tracking Time (s)')
        plt.title('Comparison of Weighted Total Tracking Time')

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom')

        ResultPlotter._save_or_show(save_path)

    @staticmethod
    def plot_radar_utilization_heatmap(bfsa_report: Dict, rule_based_report: Dict, radar_network: RadarNetwork, save_path: str = None) -> None:
        """绘制雷达通道利用率热力图"""
        radar_ids = list(radar_network.radars.keys())
        bfsa_util = np.array([bfsa_report['resource_utilization'].get(rid, 0) for rid in radar_ids])
        rule_util = np.array([rule_based_report['resource_utilization'].get(rid, 0) for rid in radar_ids])

        plt.figure(figsize=(12, 6))
        plt.imshow(np.vstack([bfsa_util, rule_util]), cmap='coolwarm', aspect='auto')
        plt.xticks(range(len(radar_ids)), radar_ids, rotation=45)
        plt.yticks([0, 1], ['BFSA-RHO', 'Rule-Based'])
        plt.colorbar(label="Channel Utilization (%)")
        plt.title('Radar Channel Utilization Heatmap')
        ResultPlotter._save_or_show(save_path)

    @staticmethod
    def plot_gantt_chart(assignments: List[csr_matrix], time_steps: List[int], mode: str, save_path: str = None) -> None:
        """
        绘制调度甘特图：
        mode = "target" 时，显示目标 vs 时间
        mode = "radar" 时，显示雷达 vs 时间
        """
        plt.figure(figsize=(12, 6))
        assignment_matrix = np.array([a.toarray() for a in assignments])

        if mode == "target":
            plt.imshow(assignment_matrix.argmax(axis=2), cmap='tab10', aspect='auto')
            plt.ylabel('Target ID')
        elif mode == "radar":
            plt.imshow(assignment_matrix.argmax(axis=1).T, cmap='tab10', aspect='auto')
            plt.ylabel('Radar ID')

        plt.xlabel('Time Step')
        plt.title(f'Scheduling Gantt Chart ({mode.capitalize()})')
        plt.colorbar(label="Assignment")
        ResultPlotter._save_or_show(save_path)

    @staticmethod
    def plot_delay_cdf(bfsa_report: Dict, rule_based_report: Dict, save_path: str = None) -> None:
        """绘制首次跟踪延迟 CDF"""
        def _compute_cdf(data):
            if len(data) == 0:
                return np.array([0]), np.array([0])
            sorted_data = np.sort(data)
            return sorted_data, np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        bfsa_delays = np.array([d for d in bfsa_report['tracking_switches'].values() if d > 0])
        rule_delays = np.array([d for d in rule_based_report['tracking_switches'].values() if d > 0])

        plt.figure(figsize=(8, 5))
        for data, label, color in zip([bfsa_delays, rule_delays], ['BFSA-RHO', 'Rule-Based'], ['#2ecc71', '#e74c3c']):
            x, y = _compute_cdf(data)
            plt.plot(x, y, marker='.', linestyle='-', label=label, color=color)

        plt.xlabel('First Tracking Delay (steps)')
        plt.ylabel('CDF')
        plt.title('Cumulative Distribution of Tracking Delay')
        plt.legend()
        plt.grid(True, alpha=0.3)
        ResultPlotter._save_or_show(save_path)

    @staticmethod
    def _save_or_show(save_path: str) -> None:
        """保存或显示图表"""
        if save_path:
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logging.info(f"Saved plot to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save plot to {save_path}: {e}")
            finally:
                plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_switch_distribution(bfsa_report: Dict, rule_based_report: Dict, save_path: str = None) -> None:
        """绘制跟踪切换次数分布（箱线图）"""
        bfsa_switches = np.array(list(bfsa_report["tracking_switches"].values()))
        rule_switches = np.array(list(rule_based_report["tracking_switches"].values()))

        plt.figure(figsize=(8, 5))
        plt.boxplot([bfsa_switches, rule_switches], labels=["BFSA-RHO", "Rule-Based"])
        plt.ylabel("Number of Tracking Switches")
        plt.title("Distribution of Tracking Switches")
        ResultPlotter._save_or_show(save_path)
