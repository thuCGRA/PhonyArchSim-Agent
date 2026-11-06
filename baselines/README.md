# baselines —— 传统算法基线（与 `photon\\\_tuner` 多智能体 DSE 对比）

本包为 `photon\\\_tuner`（LLM 多智能体 + RAG 的光子加速器 DSE）提供**可辩护、可复现**
的传统算法对比基线。它**只读**地复用 `photon\\\_tuner` 的 SimPhony 仿真器适配层、分层
搜索空间与目标函数，不修改后者任何代码。

```
baselines/
├── action\\\_space.py      # 统一动作空间
├── evaluator.py         # SimPhony 评估封装（指纹去重 + 求值计数）
├── encoding.py          # 硬件特征编码（BO 的 GP 代理模型）
├── harness.py           # 与算法无关的搜索循环（唯一差异 = propose 决策）
├── algorithms/          # rs / sa / bo / ga / sweep
├── runner.py            # 命令行：跑基线
├── compare.py           # 命令行：agent vs 基线 对比报告
└── plotting.py          # 零依赖 SVG 绘图
```

\---

## 1\. 选型依据

|基线|算法|光电方向已有工作的用法|
|-|-|-|
|**rs**|随机搜索|通用最简黑盒基线|
|**sa**|模拟退火|光子器件/电路级调优的经典方法：硅光**偏振控制器**用变步长 SA（\[arXiv:2108.08428]）；**空间光子 Ising 机**用 GA-SA 混合（\[arXiv:2605.23295]）；光网拓扑/波长分配亦常用 SA|
|**bo**|贝叶斯优化 (GP+EI)|**光子混合量子 DSE** 用「穷举 + 贝叶斯优化」（Q-PhotoMarket，\[arXiv:2610.09641]）；可编程硅光电路的**相位设置**用 BO（\[arXiv:2509.20825]）；也是当前电子加速器 DSE 的 SOTA 基线（Compass）|
|**ga**|遗传算法|**光子量子神经架构搜索**明确用「分组交叉 + 逐基因变异 + 精英保留」的 GA（Q-PhotoNAS，\[arXiv:2605.22097]）；光子 MVM 处理器 NAS（LuxNAS \[arXiv:2503.03731]、LightPro \[arXiv:2510.27013]）同属 GA/NAS 路线；光子反向设计普遍用 GA|
|**sweep**（补充）|坐标/单参数扫描|**SimPhony 原文自身的 use cases 就是逐参数扫描**（#波长、位宽），并明确把「自动化 DSE」列为 future work（\[arXiv:2411.13715]）；DxPTA（光子 Transformer 加速器 DSE）以**穷举搜索**为对比基准，自称快 15.2×（\[arXiv:2606.06515]）|

## 2\. 公平性契约（为什么这些数字可以比）

1. **同一评估器**：所有方法都走 `photon\\\_tuner.simulators.simphony.SimphonyAdapter`，
目标函数都是 `latency\\\_s × pJ\\\_per\\\_mac × area\\\_um2`（含可选 IL 预算软惩罚）。
2. **同一搜索空间**：都从 `configs/initial\\\_hardware.json` + `configs/search\\\_space.json`
出发，经 `DesignSpace` 规范化与 `validate\\\_coupling` 校验。
3. **统一动作空间**（`action\\\_space.py`）：传统算法只能做「参数移动一格 / 切换一次电路
模板 / 替换一个器件变体」的最小合法编辑，与该智能体工具箱的硬件编辑能力对齐。
因此传统算法之间、以及与传统算法和 agent 的差异**只来自决策
策略**，而非可达性。
4. **同一预算口径**：横轴统一为 **SimPhony 求值次数**。`photon\\\_tuner --iterations N`
与 `baselines --budget N` 都是 **1 次初始求值 + N 次提案**。
5. **同一产物格式**：基线的 `trajectory.json` 是 `photon\\\_tuner` 产物的**字段超集**，
因此 `compare.py` 用同一解析器读取两者。

\---

## 3\. 快速开始（real 模式，需 `.venv-simulator`）

```bash
cd /path/to/epic        # 换成你的项目克隆目录

# 单个基线：随机搜索，vit_tiny，50 次提案（共 51 次 SimPhony 求值）
.venv-agent/bin/python -m baselines.runner \\
    --algo rs --workload vit_tiny --budget 50 --seeds 0 \\
    --nn-model-factory tests.factory:build_vit_tiny

# 四个主基线 × 3 workload × 3 seed 一次跑完
.venv-agent/bin/python -m baselines.runner \\
    --algo rs,sa,bo,ga --workload vit_tiny,vit_small,vit_base \\
    --budget 50 --seeds 0,1,2 --output-root runs/run_baseline \\
    --nn-model-factory tests.factory:build_vit_tiny

# 连同补充基线 sweep
.venv-agent/bin/python -m baselines.runner \\\\
    --algo rs,sa,bo,ga,sweep --workload vit\\\_tiny --budget 50 --seeds 0,1,2
```

产物：`runs/run\\\_baseline/<algo>/<workload>/seed\\\_<N>/{trajectory.json, summary.json, iter\\\_\\\*/}`。

## 4\. 与 agent 对比

```bash
# 先有一个 agent 运行（例如 photon\\\_tuner 已在 runs/run_agent/ 下留了 run\\\_iteration\\\_\\\*）
# 再生成对比报告（表格 + 收敛曲线 SVG）
.venv-agent/bin/python -m baselines.compare \\\\
    --agent-runs "runs/run_agent/run\\\_iteration\\\_\\\*" \\\\
    --baseline-root runs/run\\\_baseline \\\\
    --out reports/compare
```

公平比较建议：把 agent 与基线放到**相同求值预算**下，例如

```bash
# agent：49 次迭代 = 50 次求值
.venv-agent/bin/python main.py --provider mock --iterations 49 --workload vit\\\_tiny
# 基线：49 次提案 = 50 次求值
.venv-agent/bin/python -m baselines.runner --algo rs,sa,bo,ga --workload vit\\\_tiny --budget 49
```

\---

## 5\. 算法参数（`--algo-option algo:key=value,...`）

|算法|选项|默认|说明|
|-|-|-|-|
|rs|`base`|`best`|随机游走基点：`best`（保留最优）/ `latest`（纯随机游走）|
|sa|`t0` / `alpha` / `t\\\_min`|`0.20` / `0.97` / `0.001`|初始温度 / 几何冷却系数 / 温度下界（在 log-objective 空间）|
|bo|`xi` / `n\\\_random`|`0.01` / `5`|EI 探索权重 / 候选集中的随机邻居数|
|ga|`pop\\\_size` / `crossover\\\_rate` / `mutation\\\_rate` / `elitism`|`12` / `0.7` / `0.3` / `1`|种群规模 / 交叉率 / 变异率 / 精英数|

例：`.venv-agent/bin/python -m baselines.runner --algo bo --algo-option bo:xi=0.05 --budget 50`

## 6\. 设计说明

* **BO** 使用 scikit-learn 的 `Matern(ν=5/2)` GP（`length\\\_scale` 固定，`n\\\_restarts=0`，
稳定且可复现），对 `-log10(objective)` 拟合，候选集 = 最优点邻域 ∪ 最新点邻域 ∪
少量随机邻居，按期望改进 (EI) 取 argmax。若环境缺 sklearn 会自动退化为随机搜索。
* **GA** 的交叉保持拓扑原子耦合（`core\\\_kind`/`node\\\_kind` 同源），器件变化由变异算子
引入。适应度 = `-log10(objective)`。
* **坐标扫描 (sweep)** 是对 SimPhony 自身「单维扫描」与 DxPTA「穷举基线」的可复现替代
（完整穷举在高维分层空间不可行）。

## 7\. 复现与确定性

* 所有随机算法通过 `random.Random(seed)` 注入，动作空间枚举顺序确定，因此同一
`(algo, workload, seed)` 会得到**逐位相同**的结果。
* 基线与智能体侧**同引擎**：均为 real 模式（真实 ONNArchSimulator）+ MCP 桥，因此求值为
**分钟级/次**，需要 `--nn-model-factory` 与 `.venv-simulator`。两者共用同一目标函数与同一份
`metrics.json`，对比才成立。
* `baselines/runner.py` 已**移除** `--mode`：仿真只走 real（analytic 解析镜像已废弃）。
