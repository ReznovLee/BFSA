#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：BFSA
@File    ：scenario_plot.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/2/5 15:30
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


# 读取 CSV 文件
def load_csv(file_path: str) -> List[Dict[str, str]]:
    """ 读取 CSV 文件并转换为字典列表 """
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


# 解析雷达数据
def process_radar_data(radar_data: List[Dict[str, str]]) -> List[Dict]:
    """ 解析雷达数据并返回雷达字典列表 """
    radars = []
    for row in radar_data:
        radars.append({
            "id": int(row["id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "radius": float(row["radius"]),
            "channels": int(row["number_channel"]),
        })
    return radars


# 解析目标数据
def process_target_data(target_data: List[Dict[str, str]]) -> Dict[int, List[Dict]]:
    """ 解析目标数据，并将相同 ID 的目标轨迹合并 """
    targets = {}
    for row in target_data:
        target_id = int(row["id"])
        if target_id not in targets:
            targets[target_id] = []
        targets[target_id].append({
            "timestep": int(float(row["timestep"])),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "vx": float(row["vx"]),
            "vy": float(row["vy"]),
            "vz": float(row["vz"]),
            "type": row["target_type"],
            "priority": int(row["priority"]),
        })
    return targets


# 可视化目标轨迹和雷达探测范围
def plot_scenario(radar_file: str, target_file: str, save_path: str = None):
    """ 绘制目标轨迹和雷达探测范围 """
    radar_data = process_radar_data(load_csv(radar_file))
    target_data = process_target_data(load_csv(target_file))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制雷达探测范围
    for radar in radar_data:
        ax.scatter(radar["x"], radar["y"], radar["z"], c='r', marker='^', label=f'Radar {radar["id"]}')
        # 画一个球形表示雷达探测范围
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = radar["radius"] * np.cos(u) * np.sin(v) + radar["x"]
        y = radar["radius"] * np.sin(u) * np.sin(v) + radar["y"]
        z = radar["radius"] * np.cos(v) + radar["z"]
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    # 绘制目标轨迹
    colors = plt.cm.jet(np.linspace(0, 1, len(target_data)))  # 给每个目标分配不同颜色
    for (target_id, trajectory), color in zip(target_data.items(), colors):
        x = [p["x"] for p in trajectory]
        y = [p["y"] for p in trajectory]
        z = [p["z"] for p in trajectory]
        ax.plot(x, y, z, label=f'Target {target_id}', color=color)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Scenario Visualization: Target Trajectories & Radar Coverage")
    ax.legend()

    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # 示例文件路径（需替换为实际路径）
    radar_csv = "scenario-2025-02-11/5-radar.csv"
    target_csv = "scenario-2025-02-11/10-targets.csv"
    save_image = "scenario-2025-02-11/scenario_visualization.png"

    plot_scenario(radar_csv, target_csv, save_image)
