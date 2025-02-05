import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from core.utils.metrics import TrackingMetrics
from scipy.sparse import csr_matrix
from core.models.radar_network import RadarNetwork

class ResultPlotter:
    """
    结果可视化工具，支持多种对比图表生成，用于分析BFSA-RHO与规则算法性能差异。
    支持的图表类型：
    1. 加权总跟踪时间对比（柱状图）
    2. 雷达通道利用率热力图
    3. 目标覆盖比例时序图
    4. 跟踪切换次数分布（箱线图）
    5. 首次跟踪延迟分布（累积分布函数图）
    """

    @staticmethod
    def plot_weighted_time_comparison(
        bfsa_report: Dict,
        rule_based_report: Dict,
        save_path: str = None
    ) -> None:
        """
        绘制加权总跟踪时间对比柱状图
        :param bfsa_report: BFSA-RHO的性能报告（来自TrackingMetrics.generate_report）
        :param rule_based_report: 规则算法的性能报告
        :param save_path: 图片保存路径（若为None则显示）
        """
        labels = ['BFSA-RHO', 'Rule-Based']
        values = [bfsa_report['weighted_total_time'], rule_based_report['weighted_total_time']]
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=['#2ecc71', '#e74c3c'])
        plt.ylabel('Weighted Total Tracking Time (s)')
        plt.title('Comparison of Weighted Total Tracking Time')
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom')
        ResultPlotter._save_or_show(save_path)

    @staticmethod
    def plot_radar_utilization_heatmap(
        bfsa_report: Dict,
        rule_based_report: Dict,
        radar_network: RadarNetwork,
        save_path: str = None
    ) -> None:
        """
        绘制雷达通道利用率热力图（对比两种算法）
        :param radar_network: 雷达网络对象（用于获取雷达坐标）
        """
        radar_ids = list(radar_network.radars.keys())
        bfsa_util = [bfsa_report['resource_utilization'].get(rid, 0) for rid in radar_ids]
        rule_util = [rule_based_report['resource_utilization'].get(rid, 0) for rid in radar_ids]
        
        # 生成雷达位置标签（简化）
        radar_labels = [f'Radar{rid}\n({r.position[0]}, {r.position[1]})' 
                        for rid, r in radar_network.radars.items()]
        
        x = np.arange(len(radar_ids))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, bfsa_util, width, label='BFSA-RHO', color='#3498db')
        plt.bar(x + width/2, rule_util, width, label='Rule-Based', color='#f1c40f')
        plt.xticks(x, radar_labels, rotation=45)
        plt.ylabel('Channel Utilization (%)')
        plt.title('Radar Channel Utilization Comparison')
        plt.legend()
        plt.tight_layout()
        ResultPlotter._save_or_show(save_path)

    @staticmethod
    def plot_coverage_timeline(
        bfsa_assignments: List[csr_matrix],
        rule_assignments: List[csr_matrix],
        time_steps: List[int],
        save_path: str = None
    ) -> None:
        """
        绘制目标覆盖比例时序图（动态变化）
        :param assignments: 各时间步的分配矩阵列表
        :param time_steps: 对应的时间步标签
        """
        bfsa_cover = [TrackingMetrics.coverage_ratio(a, a.shape[0]) for a in bfsa_assignments]
        rule_cover = [TrackingMetrics.coverage_ratio(a, a.shape[0]) for a in rule_assignments]
        
        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, bfsa_cover, label='BFSA-RHO', marker='o', linestyle='-', color='#9b59b6')
        plt.plot(time_steps, rule_cover, label='Rule-Based', marker='s', linestyle='--', color='#e67e22')
        plt.xlabel('Time Step')
        plt.ylabel('Coverage Ratio')
        plt.title('Target Coverage Ratio Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        ResultPlotter._save_or_show(save_path)

    @staticmethod
    def plot_switch_distribution(
        bfsa_report: Dict,
        rule_based_report: Dict,
        save_path: str = None
    ) -> None:
        """
        绘制跟踪切换次数分布箱线图
        """
        bfsa_switches = list(bfsa_report['tracking_switches'].values())
        rule_switches = list(rule_based_report['tracking_switches'].values())
        
        plt.figure(figsize=(8, 5))
        plt.boxplot([bfsa_switches, rule_switches], labels=['BFSA-RHO', 'Rule-Based'])
        plt.ylabel('Number of Tracking Switches')
        plt.title('Distribution of Tracking Switches')
        ResultPlotter._save_or_show(save_path)

    @staticmethod
    def plot_delay_cdf(
        bfsa_report: Dict,
        rule_based_report: Dict,
        save_path: str = None
    ) -> None:
        """
        绘制首次跟踪延迟的累积分布函数（CDF）图
        """
        def _compute_cdf(data):
            sorted_data = np.sort(data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            return sorted_data, y
        
        bfsa_delays = [d for d in bfsa_report['tracking_switches'].values() if d > 0]
        rule_delays = [d for d in rule_based_report['tracking_switches'].values() if d > 0]
        
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
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()