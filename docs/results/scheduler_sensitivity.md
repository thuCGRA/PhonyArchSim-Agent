# 调度阈值敏感性（离线回放）

- 数据来源：`runs/exp1_equal_budget/agent/seed_0/run_iteration_001/trajectory.json`
- 轮数：10
- 自检（回放是否复现真实轨迹）：**True**

调度器只有两个信号：ρ（记忆成熟度）与 β（剩余预算）。σ/π/κ 与 `single` 模式
已因「从未改变任何决策」而移除。

## 默认阈值下的轨迹

```
pipeline → pipeline → serial → serial → serial → serial → serial → serial → joint → joint
```
构成：`pipeline`×2, `serial`×6, `joint`×2

## 单阈值扫描

**rag_cold_below**

| 取值 | 模式构成 |
|---:|---|
| 0.2 | pipeline×2, serial×6, joint×2 |
| 0.34 | pipeline×2, serial×6, joint×2 |
| 0.5 | pipeline×2, serial×6, joint×2 |
| 0.67 | pipeline×4, serial×4, joint×2 |

**budget_low_below**

| 取值 | 模式构成 |
|---:|---|
| 0.1 | pipeline×2, serial×7, joint×1 |
| 0.2 | pipeline×2, serial×6, joint×2 |
| 0.3 | pipeline×2, serial×5, joint×3 |

**rag_window**

| 取值 | 模式构成 |
|---:|---|
| 1 | pipeline×2, serial×6, joint×2 |
| 2 | pipeline×2, serial×6, joint×2 |
| 3 | pipeline×2, serial×6, joint×2 |
| 4 | pipeline×2, serial×6, joint×2 |

**warm_precedent_target**

| 取值 | 模式构成 |
|---:|---|
| 1 | pipeline×2, serial×6, joint×2 |
| 2 | pipeline×2, serial×6, joint×2 |
| 3 | pipeline×6, serial×2, joint×2 |

## 判读

存在阈值取值改变了模式序列——论文中必须给出所选阈值并说明理由。
