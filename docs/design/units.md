# 指标单位

## 规范单位表

项目内部只使用一套**规范单位**（`photon_tuner/core/units.py`）：

| 键 | 物理含义 | 单位 | mobilevit 基线的量级 |
|---|---|---|---|
| `latency_ms` | 端到端延时 | ms | ~4.8 |
| `energy_mj` | 单次推理总能耗（计算 + 存储） | mJ | ~9.5 |
| `energy_compute_mj` | 光子/电器件计算能耗 | mJ | ~1.7 |
| `energy_memory_mj` | 存储能耗（HBM/GLB/RF） | mJ | ~7.8 |
| `area_mm2` | 芯片总面积（逐器件 + 节点边界框） | mm² | ~77.8 |
| `fJ_per_mac` | 每次 MAC 能耗 | fJ/MAC | ~166 |
| `area_mm2_per_mac_per_ms` | 面积 / 吞吐 | mm²/(MAC/ms) | ~5e-8 |
| `throughput_mac_per_ms` | 吞吐 | MAC/ms | ~2.2e9 |
| `core_insertion_loss` | 核心插损 | dB | ~6.2 |

## 能耗口径：计算 + 存储

`simu_energy` 只覆盖**计算**能耗（光子/电器件）；存储能耗由 `simu_memory_cost` 单独
给出。而 `simu_latency` 把两者都算进延时。若目标只取计算能耗，就会出现**计时算存储、
计价不算存储**的口径不一致——实测存储占总能耗 **75–90%**（vit_tiny 为 93%），
优化方向会系统性偏移。

因此 `metrics.energy_mj` 定义为**计算 + 存储**，两个分量另行暴露为
`energy_compute_mj` / `energy_memory_mj`，报告里还有
`memory.energy`（按存储层级，pJ）与 `energy.by_device`（按器件，pJ）。

这一改动把目标量级放大约 5–14 倍（取决于工作负载的存储占比）。这不是"数值变差"，
而是此前把 82% 的能耗排除在评价之外。

## 面积口径：逐器件 + 节点边界框

厂商 `simu_area` 的求和条件写成 `if "node" not in v`——判断的是**字典内部是否含名为
`node` 的键**，而不是键名是否含 node。后果是节点级器件（移相器/耦合器/探测器）既被
计入逐器件求和，随后节点边界框又被加了一遍：真实模式下面积被多算约 **43%**
（111.12 vs 77.75 mm²）。解析镜像的算法是正确的。

聚合层按正确口径重算 `area.total = device_area + node_bbox`，并把厂商原始值保留在
`area.vendor_total_um2`、构成明细在 `area.composition`。`area.by_chip_type` 也改为
自建分组，使**任何一个 chip 的分组都能加总到 total**（厂商版本会静默漏掉节点边界框）。

## 为什么换算系数之积为 1

目标函数是三指标之积 `J = latency × energy_per_mac × area`。上述三项换算因子分别为
`1e3`（s→ms）、`1e3`（pJ→fJ）、`1e-6`（µm²→mm²），乘积恰为 **1**，因此：

> **J 的数值在单位换算前后完全相同。**

这让单位换算是纯展示层改动：历史 run、Pareto 前沿、已发表的对比数值全部保持可比，
无需重算，也不存在"换了单位所以变好了"的歧义。这条性质由
`tests/deep/test_objective_alignment.py` 的固定锚点守住（旧锚点 433.7976 与
新基线锚点 418.7218 在同一量级，差异只来自基线网格对齐）。

## 换算发生在哪里

只有一个地方：仿真边界。

* `photon_tuner/simulators/simphony.py` 的指标组装（MCP 与非 MCP 路径共用）；
* `photon_tuner/simulators/analytic.py` 的解析镜像（离线测试用）。

此后全项目（目标函数、Pareto、工具、智能体载荷、基线、实验脚本）只处理规范单位。

## 报告里两套单位并存是**有意**的

`report.json` 同时给出：

* `metrics` —— 规范单位（ms / mJ / mm²），供 DSE 与报告引用；
* `si_metrics` —— 换算前的 SI 镜像（s / pJ / µm²），供核对与派生推导；
* `energy` / `area` / `latency` 等明细块 —— 保持 SimPhony 原生单位，
  因为这些是模拟器的逐器件输出，必须可直接追溯到它的物理模型；
* `units` —— 显式声明每个块用什么单位。

早期版本让同一个物理量在两处各自成对、却没有一处声明，是这类项目最难查的一类
bug 来源。现在任何新读取方都应先看 `units` 块，而不是凭字段名猜。

## 加权配置的兼容性

`objective_weights` 的键同时接受短名（`latency` / `energy` / `area`）、规范键与
**历史 SI 键**（`latency_s` / `energy_pj` / `area_um2`）。因此旧搜索空间配置无需改写。
