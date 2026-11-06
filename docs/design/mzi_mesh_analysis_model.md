# MZI Mesh 分析模型（phase 2，criterion 4）

> 状态：第 1 步（拓扑几何）已落地并测试；本文件锁定完整模型、器件出处与接入方案，
> 后续实现以本文为准。

## 1. 目标

把 `topology_space.py` 中 `evaluable: false` 的三个 mesh 范式
（`mzi_mesh_reck` / `mzi_mesh_clements` / `mzi_mesh_butterfly`）变成 analytic 适配器
**可求值**，使 criterion (4) 的「跨范式搜索 + Pareto 归档 + 顺序/联合消融」有第二个
真实可跑的范式。**不跑 vendored SimPhony 引擎**（real mode），只走
`photon_tuner/simulators/analytic.py` 的手写解析路径。

## 2. 拓扑几何（已实现，`simulators/mesh_model.py`）

| 范式 | MZI 数 | 深度 | 普适 | 无交叉 | 出处 |
|---|---|---|---|---|---|
| Reck | N(N−1)/2 | 2N−3 | ✓ | ✗ | 10.1103/PhysRevLett.73.58 |
| Clements | N(N−1)/2 | N | ✓ | ✓ | 10.1364/OPTICA.3.001460 |
| Butterfly | M·log₂M/2 | log₂M | ✗ | ✗ | 经典 FFT 蝶形网络(Cooley-Tukey 1965, 10.1090/S0025-5718-1965-0178586-1);光子 mesh 出处**待定** |

每个 2×2 MZI = **2 个 50:50 定向耦合器 + 2 个相移器**（内部 θ + 外部 φ）
（Clements 2016 Fig.1；Reck 1994）。这是**结构性事实**，不是器件参数。
单元测试已锁定上述计数与深度（`tests/test_mesh_model.py`，8 个断言全绿）。

## 3. 器件映射（复用已有 DOI 器件，零新造硬件）

mesh 不引入任何新器件类型，全部复用 `configs/devices/` 中**已有 DOI 出处**的变体：

| 器件 | 复用变体 | 出处 |
|---|---|---|
| 耦合器 | `LT_DC_2x2` / `ACADEMIA_DC_2x2` | （待补论文，见 §6） |
| 相移器 | `MEMS_PS` / `ITO_PS` / `IHO_PS` | nature s41378-023-00498-z / lpor.202201032 |
| 光电探测器 | `LT_PD` / `Sandia_GeWG_PD` | OE.19.024897 |
| 激光器 | `LT_Laser`（off_chip）/ `Customized_Laser` | 复用 crossbar 同一激光模型 |
| DAC/ADC/TIA | 复用 crossbar 的 `DAC_*`/`ADC_SAR_*`/`TIA_1` | 已有 DOI |

**关键原则**：mesh 的「新」只是**拓扑几何 + 器件计数 + 最长光路**，器件级数值
（IL、P_pi、面积、能耗）100% 来自上面这些已出处的器件——不编造任何新数字。

## 4. 求值模型（接入方案）

复用 `analytic.py` 现有管线（`load_required_devices → energy_calculator →
area_calculator → insertion_loss`），只需新增一个 **mesh sub_arch 构造器**，产出与
`build_sub_arch` 相同的 `sub_arch` 结构，然后整条管线无需改动：

```
build_mesh_sub_arch(hardware, catalog, kind, N):
  core.netlist.instances:
    i_mzi: [node, 0]              # scaling = N(N-1)/2 · R · C
    i_laser: [off_chip_laser, 0]  # scaling = R + C（同 crossbar）
    i_pd: [photodetector, 0]      # scaling = N · R · C（N 路输出）
    i_dac: [dac, 0]               # scaling = N · R · C（N 路输入编码）
    i_adc: [adc, 0]               # scaling = N · R · C
    i_tia: [tia, 0]               # scaling = N · R · C
  core.netlist.nets: laser → (depth 个 MZI 层) → PD
  node = tunable_mzi 网表：coupler → phase_shifter(θ) → coupler → phase_shifter(φ)
  core.il = depth · (2·coupler_IL + 2·PS_IL)
```

- **能量**：相移器 `P_pi` 按**静态功耗**计（thermo-optic 持续保持相位，`static_power =
  P_pi`，乘 `max_cycle`）——这是 mesh 与 crossbar 的本质区别（DODE 的相移器是逐操作
  动态调谐的 trimmer，mesh 的相移器是静态权重，持续耗散）。激光（复用 `_cal_laser_power`，
  IL 用 mesh 光路）+ DAC（动态输入编码）+ ADC/TIA/PD（动态读出）。**已实现**
  `weight_stationary_phase_shifter` 标志：`initialize_device_power` 据此把 `P_pi` 路由到
  `static_power`。
- **面积**：`N(N−1)/2 · (2·coupler_area + 2·PS_area)`（复用 `_node_bounding_box`）。
- **延迟**：深度 × 1/freq + DAC/ADC 调谐周期（复用 `energy_calculator`）。
- **IL**：最长光路 = `depth · MZI_IL`（MZI_IL = 2·coupler_IL + 2·PS_IL），由
  `_construct_graph`/`_longest_path` 在 node 网表上自动导出（core netlist 用
  `i_mzi(s)`/`i_mzi(e)` 接入 `replication` 链，见下）。

## 5. 正确性验证策略（“一定要确保正确性”）

1. **几何正确**（已完成）：MZI 数 / 深度 / 器件倍数与论文公式逐一对齐的单元测试。
2. **公式自洽**：mesh 的 IL 必须等于 `depth × MZI_IL`（手算 vs 图算双轨校验）。
3. **器件零编造**：mesh 求值只引用 `catalog` 里已有 DOI 的器件；加测试断言 mesh
   路径不产生任何新 `ref: ""` 的器件引用。
4. **量级 sanity**：N=4 mesh 的能量/面积/延迟与 crossbar 同量级（非 NaN/非负/非爆炸），
   Pareto 对比有意义。
5. **回归**：全量测试套件绿灯 + 离线端到端客观值改善不退化。

## 6. 待补 / 待确认（诚实标注，不编造）

- **Butterfly 出处已订正(诚实标注,零编造)**:先前的 Tang 2021(10.1021/acsphotonics.1c00419)
  出处经 Crossref 复核为「Ten-Port Unitary Optical Processor」——是 **10 端口万能酉矩阵处理器**,
  **不是**非万能的 butterfly/FFT 网格,属错误引用,已撤回。Clements 2016 全文(arXiv:1603.08788)
  检索亦无「butterfly」字样。MZI 数 `M·log₂M/2`、深度 `log₂M` 本身来自**经典 FFT 蝶形网络**
  (Cooley-Tukey 1965, 10.1090/S0025-5718-1965-0178586-1,Crosref 可解析),`ref` 现指向该 FFT
  公式出处;但**光子 mesh 的实现出处目前无可靠 DOI**,`evaluable` 维持 `false`。确认前 butterfly
  保持 `evaluable: false`。
- **耦合器 `LT_DC_2x2` / `ACADEMIA_DC_2x2` 的单器件论文**：属于「逐器件补论文」任务
  的一部分，与 criterion (5) 合并推进。
- **mesh 是否计入 DAC —— 已定案：计入（方案 A，动态输入）**。理由：mesh 做 MVM 时输入
  向量逐操作变化，输入调制器 + DAC 必须是动态的；去掉 DAC 会把 mesh 建模成「固定输入」
  的退化情形，且无法与 crossbar 在同一 workload、同一动态输入假设下做 Pareto 对比。
  同时相移器已按**静态**（持续保持）建模——这才是 thermo-optic mesh 的正确能耗故事
  （Shen 2017 里相移器是主要能耗项）。**已实现并测试**。
- **mesh 的 dataflow**：mesh 的自然 dataflow 是 `weight_stationary`（权重静态、输入流式）；
  当前沿用 `system.dataflow` 默认（`output_stationary`）。属后续细化项，不影响器件计数与 IL。

## 7. 接入顺序

1. ✅ `mesh_model.py`（几何）+ 测试。
2. ✅ `build_mesh_sub_arch` + `_tunable_mzi_node` + `evaluate` 分派（core_kind → mesh）+
   IL `replication` 链（`i_mzi(s)`/`i_mzi(e)`）+ 相移器静态功耗标志；功能测试全绿。
3. ✅ 翻 `mzi_mesh_clements.evaluable = true` + `tunable_mzi.evaluable = true` + 搜索空间
   mesh 器件表（`devices.mesh.core/node`）+ slot 校验接线 + `modify_arch_parameter` 支持
   `core_kind`/`node_kind` + `set_device_choice` 支持 `mesh_core`/`mesh_node` +
   `search_space.json` 候选 + 跨范式 Pareto 测试。
4. ✅ 端到端 mesh 搜索：`tests/deep/test_mesh_end_to_end.py` 用脚本化 solution 模型提出
   crossbar → Clements mesh 拓扑切换，orchestrator 应用→求值→Pareto 归档（`core_kind=
   mzi_mesh_clements` 点带正确 IL 级联），量级 sanity + 回归全绿。
5. ✅ Reck 已开（`evaluable=true`，出处 Reck 1994 已验证；analytic 用最坏情况深度 2N−3 的
   IL 级联，Pareto 上被 Clements 支配——同 MZI 数、更浅深度的 Clements IL 更低，物理正确）。
   Butterfly 保持 `evaluable:false`（`ref` 现指向经典 FFT 蝶形公式出处 Cooley-Tukey 1965,
   10.1090/S0025-5718-1965-0178586-1，Crossref 可解析；光子 mesh 实现出处仍待定，诚实标注，不编造 DOI）。

## 8. 顺序/联合搜索消融（criterion 4）

`DeepAgentTuner.tune(search_strategy=...)` 支持两种协同搜索模式，宿主（非 LLM）强制层顺序：

- **`joint`（默认）**：solution 每轮提出跨层 edits，全部一起应用。
- **`sequential`**：宿主按 `architecture → topology → device` 每轮只应用一层，
  其余层 edits 记为 `deferred_edits`（下一轮该层自己的迭代再提出）。

层划分（`_action_layer` / `_filter_edits_for_layer` in `runtime/orchestrator.py`）：
`core_kind`/`node_kind` = 拓扑电路层；`ARCH_PARAM_KEYS` 其余 = 架构层；`link_edits`
（`modify_device_parameter`）= 器件层。混合 `modify_hardware_parameter`（同时含架构+
拓扑 updates）会按层拆分。Pareto `meta.search_strategy` 与 `trajectory.json` 均打标，
供消融对比。测试：`tests/deep/test_sequential_joint.py`（10 例）。

