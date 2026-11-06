# 参数 × 拓扑族有效性元数据（`PARAM_SCOPE`）

> 状态：已实现。解决缺陷「架构层参数在多数拓扑下改了等于没改」。

## 1. 问题

三个网表构建器（`netlist.build_sub_arch` / `build_mesh_sub_arch` /
`build_weight_array_sub_arch`）对同一批架构参数的处理**不对称**：

| 构建器 | 对 `hardware` 的态度 |
|---|---|
| `build_sub_arch`（crossbar） | 真实读取 `core.height/width/num_wavelength/work_freq` |
| `build_mesh_sub_arch`（MZI 网格） | `height` 写成 MZI 数派生值、`width=1`、`num_wavelength=1`、`dataflow` 强制 `weight_stationary` |
| `build_weight_array_sub_arch`（WDM 阵列） | `height=n_weight`、`width=1`、`num_wavelength=n_weight`、`dataflow` 强制 `weight_stationary` |

后果：agent 在 mesh / 权重阵列拓扑上调整 `core_height`、`core_width`、
`num_wavelength`、`dataflow` 时，编辑被成功应用、闸门放行、**仿真结果逐位不变**。

real 模式实测（`.venv-simulator`，真实 torch + CACTI）：

```
MZI mesh（mzi_mesh_clements）
  baseline           E=13067512522.441687  A=6803712.0  L=0.0027834239168369997
  core_height=8      E=13067512522.441687  A=6803712.0  L=0.0027834239168369997  恒等
  num_wavelength=16  E=13067512522.441687  A=6803712.0  L=0.0027834239168369997  恒等
  work_freq=20（对照）E= 3324244476.210422  A=6803712.0  L=0.001038467916837     变化
```

由于流水线已收敛为 **real-only**（一次求值分钟级），每次无效编辑都白烧一整轮
仿真，且 agent 得不到任何反馈——所以这从「效率问题」升级为「直接烧算力」。

## 2. 两种不同性质的失效（不可混同）

原诊断把上述参数一律当成「硬编码」，这不准确。实测显示分两类：

### 类型 A：遮蔽（shadowing）——参数其实接线了，只是被更高优先级盖住

推导逻辑是「`mesh_rank` 优先，否则回退 `core.height`」
（`netlist.py` 的 `build_mesh_sub_arch` / `build_weight_array_sub_arch`）：

```python
mesh_rank = topology.get("mesh_rank")
if mesh_rank is not None:
    n_modes = int(mesh_rank)                                   # ← 赢
else:
    n_modes = int(core.get("height", core.get("width", 4)))     # ← 仅无 mesh_rank 时
```

实测验证：

```
A) mesh_rank=16 存在     core_height=8  → matrix=16   被遮蔽
                        core_height=16 → matrix=16   被遮蔽
B) 移除 mesh_rank 后     core_height=8  → matrix=8    生效
                        core_height=16 → matrix=16   生效
```

而 `default_hardware()` 永远设置 `mesh_rank=16`，所以实践中恒处于遮蔽态。

### 类型 B：无条件写死——代码里没有读取路径

```
mesh  num_wavelength = 1                    ← 字面量，无分支
mesh  dataflow       = weight_stationary    ← 无条件
权重   dataflow       = weight_stationary    ← 无条件
```

**修复策略不同**：类型 A 需要统一语义（二选一成为唯一来源），类型 B 需要么接线、
么显式声明禁用。

## 3. 四类性质

`photon_tuner/core/topology_space.py` 的 `PARAM_SCOPE` 是唯一事实来源：

| status | 含义 | 动作 |
|---|---|---|
| `effective` | 正常生效 | 可搜索 |
| `fixed_by_design` | 物理上就该固定 | 禁止搜索，`reason` 写明物理原因 |
| `superseded_by` | 被另一参数取代 | 禁止搜索，`by` 指明替代者 |
| `unsupported_by_backend` | 物理有意义、网表未接线 | 标记为可实现缺口（`implementable`） |
| `not_applicable` | 该拓扑族无此概念 | 禁止搜索 |

未出现在 `PARAM_SCOPE` 的参数默认对全部拓扑族 `effective`。

### 拓扑族划分

```python
TOPOLOGY_FAMILIES = {
    "dptc_crossbar": "crossbar_family",
    "mzi_mesh_reck": "mesh_family",      "mzi_mesh_clements": "mesh_family",
    "mzi_mesh_butterfly": "mesh_family",
    "mrr_weight_bank": "weight_array",   "pcm_crossbar": "weight_array",
    "wdm_link": "weight_array",
}
```

### 分类依据（逐项物理判定，不是一律禁止）

**`dataflow` @ mesh / 权重阵列 → `fixed_by_design`**

网格靠热光相位调谐重配置（`reconfig_time_ns = 10000`，约 10 µs），比时钟周期慢约
三个数量级，物理上不可能逐周期换权重。所以 `weight_stationary` 是**正确的物理
约束**，不是偷懒。项目里已有同类显式物理声明的先例：
`core.node.weight_stationary_phase_shifter = True`（被
`onnarchsim/workflow/energy.py` 消费）。

→ 保留物理模型不变，只补声明。

**`core_height` / `core_width` @ mesh / 权重阵列 → `superseded_by: mesh_rank`**

`mesh_rank` 同时决定矩阵维度与 MZI 数量（N=4→6, 8→28, 16→120, 32→496 个 MZI），
`core_height` 不是独立自由度。

**`num_wavelength` @ mesh → `unsupported_by_backend`（唯一真正的缺口）**

这是唯一应当「解除硬编码」而非「禁止搜索」的一项：

- **Simphony 支持 N**：四条工作流全读它
  （`workflow/utils.py` 的 `evaluate_factors` 把 `\bN\b` 替换成
  `num_wavelength`；`energy.py` / `area.py` / `insertion_loss.py` 各自读取
  `core.num_wavelength`）。
- **本项目网表没接**：mesh 的 `scaling_rules`
  （`i_laser='R+C'`、`i_mzi='R*C'`、`i_pd='16*R*C'`、`i_tia='H*W*R*C'`、
  `i_adc='(H*W)*R'`）里**一个 `N` 都没有**，且 `num_wavelength` 硬编码为 1。

物理上波长并行网格是真实架构方向，但接通它属于**架构建模变更**（需决定
PD/ADC/激光是否随 N 缩放），须单独做文献对齐 + real 验证。因此标为
`unsupported_by_backend` + `implementable: True`，并单独列入
`implementable_gaps()`，避免把这个 bug 固化成「特性」。

**`num_wavelength` @ 权重阵列 → `superseded_by: mesh_rank`**

MRR/PCM 阵列一个波长承载一个权重，`num_wavelength` 与并行权重数（`mesh_rank`）
是物理恒等关系。

**`mesh_rank` @ crossbar → `not_applicable`**

crossbar 的阵列几何由 `core_height`/`core_width` 直接给出，不存在酉网格阶数概念。

## 4. 三处消费点

### ① 搜索空间收窄

`DesignSpace.effective_candidates(param, hardware)`：参数在当前拓扑下不生效时
返回空列表，调用方据此跳过该维度。另有 `param_effectiveness()` 返回状态与建议。

```python
ds.effective_candidates("core_height", hw_crossbar)  # [8, 12, 16]
ds.effective_candidates("core_height", hw_mesh)      # []  ← 跳过
ds.effective_candidates("mesh_rank",   hw_mesh)      # [4, 8, 12, 16, 32]
```

`hardware` 省略时退化为 `candidates()`（无拓扑信息，不做收窄）。

### ② 闸门拒绝并给出可操作理由

- **architecture 阶段（第 5.4 行）**：推荐的参数必须在当前拓扑下生效，否则 fail，
  消息含 `status` / `reason` / `action`。
- **topology / solution 阶段（第 5.5 行）**：原先只校验 `_L4_PARAMS`，现扩展为
  `_L4_PARAMS ∪ _ARCH_LEGAL_PARAMS`，同样做拓扑有效性校验。

`_context_core_kind()` 的判定优先级是
**输出里的 `recommended_template`（目标结构）→ `context["hardware"]` 的当前
`core_kind`**。这样「推荐切到网格 + 同时又调 `core_height`」这类自相矛盾的组合
会按**目标结构**判失败，而不是在切换后静默失效。

实际消息示例：

```
'core_height' has no effect under MZI 网格 (core_kind='mzi_mesh_clements'):
it is superseded by 'mesh_rank' — 网格的矩阵维度由拓扑层 mesh_rank 决定，
height 只是其派生量（MZI 总数 = 2N(N-1)），两者描述同一物理量且 mesh_rank 优先。
Suggested: 网格阶数 N 改由 topology 阶段的 mesh_rank 控制；架构阶段请改用
tiles / cores_per_tile / work_freq 等生效参数.
```

### ③ Agent 可主动查询

新增工具 `searchable_parameters`（`toolbox/bindings/topology_tools.py`），返回：

```json
{
  "core_kind": "mzi_mesh_clements",
  "family": "mesh_family",
  "searchable": ["mesh_rank"],
  "disabled": {"core_height": {"status": "superseded_by", "by": "mesh_rank",
                               "reason": "...", "action": "..."}, ...}
}
```

已加入 `ARCHITECTURE_TOOLS` / `TOPOLOGY_TOOLS` / `SOLUTION_TOOLS` 三个白名单
（管线工具总数 27 → 28），并在 prompt 中要求**推荐前先调用**。

## 5. 顺带修复的真实缺陷

`_run_architecture_stage` 的 `stage_context` 原先只有
`{"candidate_layers", "metrics"}`——**没有 `hardware` / `design_space` / `mode`**。
导致：

1. 架构闸门拿不到拓扑信息，新校验会退化为 crossbar 家族判断（要么漏检、要么误判）；
2. `architecture_legality_check` 里原有的「推荐值必须是枚举候选」检查因
   `design_space is None` 而**完全没生效**（死代码）。

已在 `orchestrator.py` 补上这三个字段。修复后该值枚举检查同时复活：

```
'work_freq=999' -> fail: not an enumerated candidate; legal: ['5','10','15','20','25']
```

## 6. 防漂移护栏

`tests/test_param_scope.py`（24 项）分两层：

- **声明层**：四类性质未被混同（如 `dataflow` 必须是 `fixed_by_design` 且不带
  `implementable`；网格 `num_wavelength` 必须是 `unsupported_by_backend` 且
  `implementable=True`；`superseded_by` 必须指明存在的替代参数；所有非
  `effective` 状态必须带 `reason` + `action`）。
- **事实层**：对每个 `(参数, core_kind)` 组合实际调用构建器，对决定仿真结果的字段
  取指纹后比对——声明 `effective` 必须变、声明非 `effective` 必须不变。
  **新增拓扑或改动构建器却忘了更新 `PARAM_SCOPE` 时，这条测试会失败。**

## 7. 验证状态

- 主测试套件 `325 tests OK`；`tests/deep` `118 tests OK`
- 端到端 real 模式 CLI 烟雾测试通过（`metrics.json` 产出
  `energy_pj` / `area_um2` / `latency_s`，exit 0）
- 元数据与实测 8×12 有效性矩阵逐格一致

## 8. 后续可做

- 接通网格的 `num_wavelength`（唯一标记为可实现的缺口）：需为 mesh 设计 N 的器件
  缩放规则，并重新验证 `scaling_rules` 在 real 模式下的一致性。
- 把 `PARAM_SCOPE` 与 `configs/search_space.json` 打通（当前 JSON 只描述候选值，
  不含拓扑有效性）。
