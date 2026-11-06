# 原生 SimPhony 拓扑扩展：按论文建模方法评估论文全部拓扑

> 状态：已实现并测试（301 域测试全绿）。本文锁定扩展的建模依据、代码位置与诚实性约束。

## 1. 依据：论文建模方法（arXiv:2411.13715）

论文的架构建模是 **DevLib（实验数据器件功率模型）+ Arch（层次化参数化网表，instances/
nets/scaling_rules/replication/ports_num，节点=最小积木，加权 DAG 自动导出关键路径）+
Sim（ONN 转换 + 工作负载提取 + 多维并行 + 周期延时 + 内存层次）**。本次扩展逐条落实其
定量模型：

| 论文机制 | 落地位置 | 说明 |
|---|---|---|
| 关键路径 IL（加权 DAG 最长路径） | `analytic.py::insertion_loss` | 已存在；本次为 Reck/butterfly 注入 crossing |
| 链路预算 eq.(1) | `analytic.py::_link_budget` | `P_laser = 10^((S+IL)/10)·2^bin/(η_WPE·(1−0.1^(ER/10)))`；SNR 由 `2^bin` 幅度余量隐含 |
| I 因子（Tab. I range forwards） | `energy_calculator` 读 `core.range_forwards` | MRR=2、PCM=4、MZI/butterfly/TeMPO/WDM=1 |
| 重配置延时惩罚 | `energy_calculator` 读 `core.reconfig_time_ns` | `τ_tot=τ_load+τ_write+I(τ_comp+τ_reconfig)`；TO=10µs、PCM=100ns、500 cycles/switch@5GHz |
| weight-stationary 数据流限定 | `build_mesh_sub_arch` / `build_weight_array_sub_arch` | 静态 MZI / MRR / PCM / 微梳权重均为 weight-stationary |

## 2. 已支持拓扑（7 core × 5 node）

| core_kind | node_kind | I | 重配置 | 普适/相干 | 出处 |
|---|---|---|---|---|---|
| `dptc_crossbar` | `dode` | 1 | 0（动态） | ✓/✓ | Hamerly 2019 |
| `mzi_mesh_reck` | `tunable_mzi` | 1 | 10µs（TO） | ✓/✓ | Reck 1994 |
| `mzi_mesh_clements` | `tunable_mzi` | 1 | 10µs（TO） | ✓/✓ | Clements 2016 |
| `mzi_mesh_butterfly` | `tunable_mzi` | 1 | 10µs（TO） | ✗/✓ | Cooley-Tukey 1965（光子 mesh 出处待定） |
| `mrr_weight_bank` | `weight_mrr` | 2 | 10µs（TO） | ✓/✗ | Tait 2017（10.1038/s41598-017-07754-z） |
| `pcm_crossbar` | `pcm_cell` | 4 | 100ns（PCM） | ✓/✗ | Feldmann 2021（10.1038/s41586-020-03070-1） |
| `wdm_link` | `wdm_modulator` | 1 | 0（动态） | ✓/✓ | Xu 2021（10.1038/s41586-020-03063-0） |

`real` 路径（torchonn ONNArchSimulator）对全部 7 个拓扑可用：mesh 用标准 MZI-mesh 网表、
权重阵列用 `build_weight_array_sub_arch` 的通用网表（微梳 + 权重/调制单元 + PD），`range` 字段以
`loading_factor_calculator` 的枚举（`full`/`positive`）表达（MRR=positive/full/full→I=2、
PCM=positive/positive/full→I=4、WDM=full/full/full→I=1），经 `.venv-simulator` 端到端跑通。
**TeMPO 不是独立拓扑。** 它曾以 `tempo_crossbar` 注册，但复用 `dptc_crossbar` 的 crossbar 网表，
只改 `name`/`type`/`ref` —— 网表与 energy/area/latency 全部逐位相同。它唯一的"差异"（时域复用）
本就是 `dptc_crossbar` 可搜索的 `temporal_accum` 轴，慢光 EOM 则是 device 维度的 `SL_MZM` 选型。
因此它是**数值与表达能力双重冗余**的，已删除以免复制一整份逐位相同的评估结果；
`tests/test_topology_duplicates.py` 机械校验此类重复不会复发。

## 3. 关键建模决策与诚实性

1. **crossing 建模**（Reck/butterfly）：`mesh_model.py::n_crossing` 给出结构级交叉数（非器件参数）；
   `build_mesh_sub_arch` 仅在 `crossing_free=False` 时注入 `i_cross` 实例（默认 `WG_Crossing_Parabolic`，
   DOI 10.1364/OL.32.002801），串入 MZI 链之后、PD 之前 → 关键路径额外累计 `depth×IL_crossing`，
   面积按总交叉数缩放。Clements 无交叉，不注入。
2. **butterfly 启用**：结构事实（MZI 数 `M·log₂M/2`、深度 `log₂M`）源自 Cooley-Tukey 1965 +
   Clements/Reck 的 2×2 MZI 分解；**无单一光子 mesh DOI**（Tang-2021 归属已撤回）。分析可求值
   （器件数/深度/IL/面积/延时）；`real=True` 以标准 MZI-mesh 网表表达其结构（非专用光子 FFT 网表），
   不冒充光子 FFT 来源。
3. **并行权重阵列**（MRR/PCM/WDM）：光**一次通过**一个权重单元（drop/吸收/调制），节点网表为单器件、
   无内部连线；`n_weight` 只经缩放规则计入器件数量（面积/能耗），不进入插损关键路径。故 IL 低
   （0.3–1.2dB）而面积随权重数增长——与串行 MZI mesh（深度放大 IL）形成真实权衡。
4. **器件扩展**（2 个新类型，各 1 真实 + 1 折中构造）：
   * `weight_mrr`：`WG_MRR_WEIGHT_TAIT`（traceable，Tait 2017 DOI）+ `WG_MRR_WEIGHT_LP`（baseline，折中构造）。
   * `pcm_cell`：`PCM_CELL_FELDMANN`（traceable，Feldmann 2021 DOI）+ `PCM_CELL_LP`（baseline，折中构造）。
   折中构造变体 `ref:""` 并在 `BASELINE_PROVENANCE_NOTES` 逐条注明“constructed (no single paper)”，
   绝不冒充论文实测值。
5. **权重阵列 = 单输出点积（向量）**：本实现把 `mesh_rank` 解释为并行权重数（WDM 通道数），
   三类拓扑统一为“单神经元点积”。这是对 PCM 交叉开关（2D 阵列）与 WDM 链路的一阶结构建模；
   `matrix_height=1` 明确标记其向量语义。**吞吐归一化已实现**：工作负载循环按
   `[tiles, cores, matrix_height, matrix_width]` 重算（`workload_arch_context` + `build_workload`），Pareto 归档轴
   改为 `pJ_per_mac`（每 MAC 能耗）+ `area_um2_per_mac_per_s`（单位吞吐面积）+ `latency_s` +
   `core_insertion_loss`，使向量与矩阵阵列在吞吐口径下可比（见 §5）。
6. **TeMPO 不注册为独立拓扑**：它的时域复用是 `dptc_crossbar` 的 `temporal_accum` 轴、
   慢光 EOM 是 device 维度的 `SL_MZM` 选型，二者都无需新 core_kind。曾用 `tempo_crossbar`
   注册，实测网表与三项指标与 `dptc_crossbar` 逐位相同，属纯冗余，已删除。

## 4. 代码位置

* 拓扑几何：`photon_tuner/simulators/mesh_model.py`（`n_crossing`、`range_forwards`）。
* 解析镜像：`photon_tuner/simulators/analytic.py`
  （`_link_budget`、`_weight_element_node`、`_WEIGHT_ARRAY_SPECS`、`build_weight_array_sub_arch`、
  `energy_calculator` 的 I 因子/重配置、`build_mesh_sub_arch` 的 crossing/weight-stationary）。
* 注册表：`photon_tuner/core/topology_space.py`（`CORE_KINDS`/`NODE_KINDS`/`COUPLING_MATRIX`/
  `REQUIRED_DEVICE_SLOTS`/`WEIGHT_ARRAY_CORE_KINDS`）。
* 器件表：`photon_tuner/core/search_space.py`（`DEFAULT_WEIGHT_ARRAY_*_DEVICES`）。
* 器件库：`configs/devices/photonics/{weight_mrr,pcm_cell}.yml`；溯源注记
  `photon_tuner/core/provenance.py::BASELINE_PROVENANCE_NOTES`。
* 搜索空间：`configs/search_space.json` 拓扑层（core_kind 8 值、node_kind 5 值）。

## 5. 已知局限（保留，不掩盖）

* real 路径需 `.venv-simulator`（Python 3.10 + torchonn/pyutils）；8 个拓扑均已端到端跑通。
* 吞吐归一化用各工作负载自身的矩阵尺寸（N/D/M）按 miniblock 重算 iter；`mesh_rank` 默认 4 的小
  mesh 与 12×12 crossbar 的吞吐差异由 `matrix_height/width` 捕获，但 expressivity（酉通用性 / R+
  值域）仍只记录在元数据，不进入 Pareto 数值轴。
* mesh 仍 `num_wavelength` 硬编码 1；`matrix_height/width` 已被 `workload_arch_context` 消费为
  miniblock（与真实 ONNArchSimulator 的 `arch_blocks` 一致）。
* `devices.mesh` / `devices.weight_array` 与 `devices.core/node` 为并列器件表（非统一 schema）。
* 器件 YAML 位于 vendored 树内（`VENDORED.md` 标注“只读引用”）；升级会整体覆盖新增变体。

## 6. DOI 验证（Crossref，2026-10 复核）

| DOI | 出处 | Crossref 解析 |
|---|---|---|
| 10.1038/s41598-017-07754-z | Tait 2017，硅光子权重库 | ✓ HTTP 200，“Neuromorphic photonic networks using silicon photonic weight banks” |
| 10.1038/s41586-020-03070-1 | Feldmann 2021，集成光子张量核 | ✓ HTTP 200，“Parallel convolutional processing using an integrated photonic tensor core” |
| 10.1038/s41586-020-03063-0 | Xu 2021，11 TOPS 光子卷积加速器 | ✓ HTTP 200，“11 TOPS photonic convolutional accelerator for optical neural networks” |

折中构造变体（`WG_MRR_WEIGHT_LP`、`PCM_CELL_LP`）`ref:""` 为 baseline，无 DOI 可解析，如实记录。
