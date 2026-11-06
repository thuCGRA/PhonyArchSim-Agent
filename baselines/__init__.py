"""传统（非 LLM）优化算法基线，用于与 ``photon_tuner`` 的 LLM 多智能体 DSE 对比。

本包**只读**地复用 ``photon_tuner`` 已提供的仿真器适配层、搜索空间模型与目标
函数（不修改其任何代码），并让所有传统算法共用同一「统一动作空间」与同一条
SimPhony 评估路径，从而保证对比实验的差异**只来自决策策略**，而非可达性、评估
器或目标函数。

对外入口：

- :mod:`baselines.action_space` —— 统一动作空间（保证各方法可达性一致的公平性契约）
- :mod:`baselines.evaluator`  —— SimPhony 评估封装（指纹去重 + 求值计数）
- :mod:`baselines.encoding`   —— 硬件特征编码（供 BO 代理模型使用）
- :mod:`baselines.harness`    —— 与算法无关的搜索循环
- :mod:`baselines.algorithms` —— RS / SA / BO / GA + 可选坐标扫描
- :mod:`baselines.runner`     —— 命令行入口
- :mod:`baselines.compare`    —— 与 agent 运行结果的对比报告
- :mod:`baselines.plotting`   —— 零依赖 SVG 绘图
"""

from __future__ import annotations

__version__ = "0.1.0"