# Photonic DSE 搜索空间（修订版 v2）

> 基于 SimPhony 实际代码（`onnarchsim/workflow/*`、`onnarchsim/database/*`）与
> `configs/design/**`、`configs/devices/**` YAML 核对后修订。
> 每行参数标注真实落地状态：
> - **✅ 已生效**：当前已暴露且能改变仿真结果
> - **⚠️ real-only**：只在 real(CACTI) 模式生效，analytic 镜像忽略
> - **🔧 已有机制未暴露**：SimPhony 原生支持，但当前 search space 未开放
> - **🆕 需新增模型**：SimPhony 无此模型，加参数不会生效，需先扩仿真器

---

## 0. 事实基线（与 GPT 版的关键差异）

| 项目 | GPT 版说法 | 实际 |
|---|---|---|
| 器件类别数 | 13 类 | 库 18 类；13 类为当前网表已接线 |
| 异构形态 | tile_groups / group_* | 每个 layer type → 一个 sub_arch（多 sub_arch 异构） |
| L5.2 字段 | MZM 驱动电压 / TIA 增益带宽 / PD 暗电流 / MRR 谐振特性 / WG 长度 | 这些字段**不存在**，以 YAML 实际字段为准 |
| GLB/RF cache_size | 普通「已有」参数 | **real-only**，analytic 模式不生效 |
| matrix_tiling / loop_order / layer_mapping / scaling_rules / accumulation | 全部「待扩展」 | 多数是「已有机制未暴露」 |
| 跨层约束 | 未提及 | 3 个硬约束已存在（见 §7） |

---

## 1. 六层结构（保留 GPT 的工程分类，只修正内容）

L1 系统架构 · L2 存储与数据映射 · L3 光学计算核心 · L4 拓扑与电路 · L5 光电器件 · L6 精度与运行

六层是**工程分类**，不是 SimPhony 官方层级；SimPhony 的配置本体是「sub_arch → core(Dptc) → node(Dode)」三层嵌套。

---

## 2. L1 系统架构层

| 子空间 | 参数 | 候选 | 状态 |
|---|---|---|---|
| 1.1 计算资源规模 | tiles | 1/2/4/8 | ✅ 已生效 |
| 1.1 计算资源规模 | cores_per_tile | 1/2/4 | ✅ 已生效 |
| 1.2 异构组织 | sub_arch_count | 1..N | 🔧 已有机制未暴露 |
| 1.2 异构组织 | layer→sub_arch 映射 | QConv2d/QLinear/QMatMul → sub_arch_X | 🔧 已有机制未暴露 |
| 1.2 异构组织 | 每 sub_arch 独立配置 | tiles/cores/几何/精度/dataflow/memory 各自一套 | 🔧 已有机制未暴露 |

**说明（修正）**：异构的正确语义是「不同 layer type 映射到不同 sub_arch」，由
`LT_hetero.yml` 的 `sub_archs` + `architecture_mapping/*.yml` 驱动；当前只有 1 个
sub_arch（LT），所以现在是同构。**不要**引入 SimPhony 不存在的「tile_group」概念。

---

## 3. L2 存储与数据映射层

| 子空间 | 参数 | 候选 | 状态 |
|---|---|---|---|
| 2.1 存储资源 | hbm_bandwidth | 512/1024/2048 GB/s | ✅ 已生效 |
| 2.1 存储资源 | hbm_read/write_energy | 62.4（固定） | 🔧 已有机制未暴露 |
| 2.1 存储资源 | glb1/glb2/rf cache_size | 64/256/485/1024/4096 | ⚠️ real-only |
| 2.1 存储资源 | glb1/glb2/rf block_size | 8（固定，bus_width=block_size*8） | 🔧 已有机制未暴露 |
| 2.2 数据流 | dataflow | output/weight/input_stationary | ✅ 已生效 |
| 2.2 数据流 | buffer_allocation | — | 🆕 需新增模型 |
| 2.2 数据流 | double_buffering | — | 🆕 需新增模型 |
| 2.2 数据流 | prefetch_distance | — | 🆕 需新增模型 |
| 2.3 工作负载映射 | dim_map (M/N/D→width/height/wavelength) | 枚举 | 🔧 已有机制未暴露 |
| 2.3 工作负载映射 | matrix_tiling（≈miniblock+dim_map） | — | 🔧 已有机制未暴露 |
| 2.3 工作负载映射 | loop_order（≈dataflow） | 3 种 | 🔧 已有机制未暴露 |
| 2.3 工作负载映射 | layer→sub_arch 映射策略 | — | 🔧 已有机制未暴露 |

**说明**：SimPhony 的 buffer 大小由 `generate_memory_setting()` 按 dataflow **自动推导**，
不存在独立的 buffer_allocation 参数；double_buffering/prefetch 完全没有对应模型。

---

## 4. L3 光学计算核心层

| 子空间 | 参数 | 候选 | 状态 |
|---|---|---|---|
| 3.1 计算阵列规模 | core_height | 8/12/16 | ✅ 已生效 |
| 3.1 计算阵列规模 | core_width | 8/12/16 | ✅ 已生效 |
| 3.1 计算阵列规模 | op_type | MVM/VV/MM | 🔧 已有机制未暴露 |
| 3.2 光学并行 | num_wavelength | 8/12/16 | ✅ 已生效 |
| 3.2 光学并行 | work_freq | 5/10/15/20/25 GHz | ✅ 已生效（受 §7 约束） |
| 3.2 光学并行 | range (weight/input/output = full/complex/positive) | 枚举 | 🔧 已有机制未暴露 |
| 3.2 光学并行 | wavelength_allocation | — | 🆕 需新增模型 |
| 3.3 外围资源共享 | dac/adc/tia 等 sharing | = netlist.scaling_rules 表达式 | 🔧 已有机制未暴露（见 L4） |
| 3.3 外围资源共享 | readout_multiplex_factor | — | 🆕 需新增模型 |

**说明（修正）**：资源共享在 SimPhony 里不是「开关字符串」，而是 netlist 的
`scaling_rules` 表达式（如 ADC `(H*W)*R`、TIA `H*W*R*C`）。要改共享，必须改 YAML 里的
表达式并保证与拓扑一致——GPT 这点说对了，但它应归为 L4 的 device_scaling_rules 而非 L3 的独立开关。

---

## 5. L4 光学拓扑与电路层

| 子空间 | 参数 | 状态 |
|---|---|---|
| 4.1 拓扑结构 | core.type（Crossbar；MZI Mesh 有部分计数/布局钩子） | 🔧 已有机制未暴露 |
| 4.1 拓扑结构 | topology_template | 🆕 需新增模板 |
| 4.2 节点连接 | netlist.instances / nets | 🔧 已有机制未暴露（当前固定） |
| 4.2 节点连接 | netlist.scaling_rules（器件实例数缩放） | 🔧 已有机制未暴露（字符串表达式） |
| 4.2 节点连接 | netlist.replication / temporal_accum_factor | 🔧 已有机制未暴露 |
| 4.3 光路设计 | insertion_loss 最长路径（由 nets 自动算） | ✅ 已生效（器件选择会影响） |
| 4.3 光路设计 | accumulation_strategy（temporal_accum_factor + range） | 🔧 已有机制未暴露 |
| 4.3 光路设计 | optical_splitting | 🆕 需新增模型 |

**说明**：MRR Array、Butterfly **没有任何**支持（连钩子都没有）；MZI Mesh 有
`clements_mzi_*_counter` 和注释掉的 bounding-box 实现，属于半支持。建议严格从
Crossbar 模板起步。

---

## 6. L5 光电器件层

### 6.1 器件型号选择

器件库共 **18 类**，当前网表接线 **13 类**（`dptc.yml` 9 + `dode.yml` 4）。
未接线的 5 类：`integrator / crossing / large_insertion / mmi / off_chip_laser`。

| 器件类型 | 库内型号数 | 当前接线型号 | 状态 |
|---|---|---|---|
| mzm | 4（ACADEMIA_EO_MZM_2x2/1x1, SL_MZM, LT_MZM） | LT_MZM | ✅ 可搜索 |
| dac | 2（DAC_1=12bit, DAC_2=8bit） | DAC_2 | ✅ 可搜索 |
| adc | 1（ADC_SAR_1=8bit SAR） | ADC_SAR_1 | ✅ 可搜索 |
| tia | 1 | TIA_1 | ✅ 可搜索 |
| waveguide | 1 | WG_Single_Strip | ✅ 可搜索 |
| mrr_rerouter | 1 | LT_MRR_REROUTER | ✅ 可搜索 |
| laser_splitter | 1 | Customized_Laser_Splitter | ✅ 可搜索 |
| micro_comb | 1 | LT_Micro_Comb | ✅ 可搜索 |
| on_chip_laser | 2（LT_Laser, Customized_Laser） | Customized_Laser | ✅ 可搜索 |
| phase_shifter | 4 | MEMS_PS | ✅ 可搜索 |
| photodetector | 3 | LT_PD | ✅ 可搜索 |
| coupler | 4 | LT_DC_2x2 | ✅ 可搜索 |
| y_branch | 2 | LT_Y_Branch | ✅ 可搜索 |
| integrator | 1 | 未接线 | 🆕 需改网表 |
| crossing | 5 | 未接线 | 🆕 需改网表 |
| large_insertion | 1 | 未接线 | 🆕 需改网表 |
| **mmi** | 1（Customized_1x10_MMI） | **已接线**（splitter_kind=mmi 时 i6 实例） | ✅ 可搜索（实例切换） |
| **off_chip_laser** | 1（LT_Laser） | **已接线**（laser_source=off_chip 时 i4 实例） | ✅ 可搜索（实例切换） |

> 更新：`mmi` 与 `off_chip_laser` 已作为**实例级器件类型切换**进入搜索空间（device 层
> `splitter_kind` / `laser_source` 两个 enum 参数）：切换时保持实例编号/缩放规则/光路位置
> 不变，只换器件类型。mmi 走 SimPhony 原生的 `cal_MMI_area`（需 `netlist.ports_num`），
> off_chip_laser 走 laser 功率模型且 cfgs 与 on-chip 物理可区分（wall_plug_eff 0.3、
> +2 dB 光纤耦合损耗）。消融验证：`laser_source=off_chip` → energy −2.75%；
> `splitter_kind=mmi` → area +0.13%、energy +0.79%（vit_tiny，real 模式，
> `runs/evidence_instance/evidence.jsonl`）。

### 6.2 器件物理参数（以 YAML 实际字段为准，当前应保持 fixed）

| 器件 | 真实可读字段 | 状态 |
|---|---|---|
| MZM | efficiency(fJ/bit), modulation_speed, testing_bit, eo_bandwidth, extinction_ratio, insertion_loss, length, width, static_power | fixed（**无「驱动电压」**） |
| DAC | tech, area, prec, power, sample_rate, FoM | fixed |
| ADC | tech, type, area, prec, power, sample_rate | fixed |
| TIA | tech, area, dynamic_power | fixed（**无「增益/带宽/噪声」**） |
| PD | responsivity, bandwidth, sensitivity, static_power(暗电流代理) | fixed |
| MRR | static_power(调谐功耗), length, width, insertion_loss | fixed（**无「谐振特性」**） |
| Laser | power, wall_plug_eff(转换效率), length, width, (insertion_loss) | fixed |
| Waveguide | mode, width, type, insertion_loss(dB/cm) | fixed（**无「长度」**） |

**说明**：当前 `design_space.py` 已把 device_param 层标为 `fixed`（只选型号、不覆盖 cfgs），
这是对的。未来若开放物理参数，必须让「参数名 = YAML 字段名」一一对应。

---

## 7. 跨层可行性约束（GPT 版缺失，需补进搜索空间）

这三个约束已存在于 `DesignSpace.violations()` / `normalize()`：

1. `work_freq ≤ MZM.modulation_speed / MZM.testing_bit`（LT_MZM = 25 GHz）
2. `precision.in_bit ≤ DAC.prec`（DAC_2 = 8）
3. `precision.out_bit ≤ ADC.prec`（ADC_SAR_1 = 8）

含义：work_freq/精度/器件型号**不能独立搜索**，任何候选组合都要过约束；器件换成低性能型号时，
work_freq 与精度的合法候选会被压缩。

---

## 8. L6 精度与运行层（修正编号）

| 子空间 | 参数 | 候选 | 状态 |
|---|---|---|---|
| 6.1 全局精度 | in_bit / w_bit / out_bit | 2/4/8 | ✅ 已生效（受 §7 约束） |
| 6.2 混合精度 | 每层精度 | — | 🆕 需新增模型（当前 nn_mapping 每层也是统一值） |
| 6.2 混合精度 | 累加精度 | — | 🆕 需新增模型 |
| 6.3 动态运行 | active_core_count / frequency_schedule / laser_activation | — | 🆕 需新增模型 |
| 6.4 重配置 | weight_reconfiguration / batching | — | 🆕 需新增模型 |

---

## 9. 统计汇总（修正）

| 层级 | 已生效 | 已有机制未暴露 | 需新增模型 |
|---|---|---|---|
| L1 系统架构 | 2 | 3 | 0 |
| L2 存储与映射 | 2 | 5 | 3 |
| L3 光学核心 | 4 | 4 | 2 |
| L4 拓扑与电路 | 1 | 5 | 2 |
| L5 光电器件 | 13 类（型号） | 0 | 5 类接线 |
| L6 精度与运行 | 3 | 0 | 6 |
| 合计 | 12 生效参数 + 13 类型号 | 17 项 | 13 项 |

> 统计口径：只数「能改变仿真结果」的才算生效；real-only 的 3 个 cache_size 记为
> 「已有机制未暴露」（实际是 real-only 未暴露，单列亦可）。

---

## 10. 落地建议（对齐现有代码）

项目**已有**分层实现 `photon_tuner/core/design_space.py`（4 层：
system / optical_core / device / device_param），不要推翻。建议按以下顺序增量落地：

1. **先把「已有机制未暴露」的 17 项暴露成参数**（性价比最高，零仿真器改动）：
   - L1：sub_arch 异构（多 sub_arch + layer 映射）
   - L2：dim_map、block_size、hbm read/write energy
   - L3：op_type、range(full/complex/positive)
   - L4：scaling_rules（共享策略，字符串表达式受控白名单）
2. **给 real-only 参数显式打标**（现有 `mode:"real"` 已做，保持并在文档注明）。
3. **在 search space 里内建 §7 三个跨层约束**（`violations()` 已实现，接入搜索循环即可）。
4. **「需新增模型」的 13 项按依赖排序**，先做有物理模型可依的（如 integrator 累加、
   off_chip_laser 切换），后做纯策略类（prefetch/reconfiguration）。
5. 目录结构建议沿用「层级 → 子空间 → 参数」三级，但每行强制带 `status ∈ {live, real-only,
   unexposed, new-model}` 字段，避免再次把「无中生有」和「已存在」混为一谈。
