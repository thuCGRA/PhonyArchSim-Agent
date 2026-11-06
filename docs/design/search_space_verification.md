# 搜索空间有效性验证机制 — 实现方案（待评审）

> 状态：设计草案。目的：先明确「怎么建、建哪些文件、每条证据长什么样」，
> 评审通过后再写代码。核心结论——\*\*Workload 是贯穿全流程的条件，不是最后
> 追加的一次智能体分析\*\*；空结果（参数改了、指标没动）是一等公民证据。

\---

## 1\. 目标

建立一套证据系统，回答：

> 某个硬件参数 \*\*P\*\*，在什么工作负载 \*\*W\*\* 下，通过什么硬件映射机制
> （`param → hardware → sub_arch → simulator`），产生了怎样的性能变化
> （`energy/area/latency/objective` 各自怎么动），以及这一变化\*\*是否可信\*\*。

它必须能区分四类结果：

|结果|含义|是否可信证据|
|-|-|-|
|生效|注入确认 + 指标变化|✅ 正向证据|
|**空结果 (null)**|注入确认 + 指标不变|✅ 反向证据（该参数对该负载该指标无效）|
|映射断裂|注入未确认（值没落到 sub_arch）|❌ 代码 bug，需修|
|约束截断|参数被 `normalize`/`violations` 夹回|⚠️ 记录 clamp 前的值|

\---

## 2\. 设计原则

1. **单变量消融**：每次只改一个参数，其余保持基线，杜绝混叠。
2. **注入回读是硬门槛**：每个扰动后必须回读 `simulator.sub_arch_dictionary[name]`
并断言新值落在目标路径，否则该条证据直接标记 `injection_confirmed=false`。
3. **确定性让 null 可信**：仿真器位级确定（已实测），所以「改了参数指标没动」
是真实 null，不是噪声。
4. **拆原始指标，不只 objective**：目标函数 `energy×area×latency` 中 **memory
energy 不在 objective**（energy/area 是计算-only，latency 含 memory），所以
必须记录原始三指标 + objective + 单独的 memory_energy，否则 `cache_size`
类参数会被误判为无效。

\---

## 3\. 验证流程（8 步具体化）

对每个 `(参数 P, 候选值 V, 工作负载 W)` 组合：

```
1. 取基线 hardware B，构造 B' = step/modify(P → V)
2. 合法性/约束检查：DesignSpace.violations(B') + normalize(B')
   → constraint_result ∈ {pass, clamped, violated}（记录 clamp 前后值）
3. 映射到 hardware：DesignSpace.set(B', P, V)（走 map 路径，已被 step/modify 完成）
4. 注入 SimPhony：_inject_hardware(simulator, B', simphony_root)
5. 确认仿真器确实读取：回读 sub_arch_dictionary[name] 在 SUB_ARCH_PARAM_MAP[P]
   路径上的值 == V  → injection_confirmed
6. 跑仿真 → report.json：energy / area / latency / memory_energy / objective
7. 与基线逐项比：metric_deltas{energy%,area%,latency%,objective%} +
   affected_metrics[] + is_null
8. 写证据：EvidenceStore.record(一条 ablation 记录)
```

每一步都携带 `workload` 与 `workload_features`（见 §7），因此不是「跑完再加
一次负载分析」，而是负载作为条件贯穿 2/3/5/6/7/8 每一环。

\---

## 4\. 三层映射链与「注入回读」机制（本方案的关键）

现有代码里同一个参数有三种表示，证据系统必须显式记录并校验整条链：

|层|例：work_freq|例：laser_distribution|
|-|-|-|
|1. 搜索参数|`work_freq`|`laser_distribution`|
|2. hardware 路径（`DesignSpace.map`）|`["arch","optical_core","work_freq"]`|`["arch","topology","laser_distribution"]`|
|3. sub_arch 路径（仿真器真正读的）|`["core","work_freq"]`|`["core","netlist","laser_power_distribution"]`|

**问题**：第 3 层目前只**隐式**存在于 `build_sub_arch`（手动写 `core_cfg["work_freq"]`
等），没有显式表。→ 本方案新增 `SUB_ARCH_PARAM_MAP`（放在
`photon_tuner/simulators/netlist.py`，紧挨 `build_sub_arch`），并加一个单元测试断言
`build_sub_arch` 的输出与这张表一致，防止表漂移。

> 注：网表装配已从 `analytic.py` 拆出到 `simulators/netlist.py`（`analytic.py` 现仅保留
> 求值数学），因此映射表归属 `netlist.py`。

**注意拓扑参数的特殊性**：`sharing_policy / temporal_accum / laser_distribution`
不是标量，而是经 `apply_topology` 重塑 `core.netlist`（如
`netlist["temporal_accum_factor"]["duration"]`、`netlist["laser_power_distribution"]`、
sharing 影响 `scaling_rules`/`instances`）。它们的回读目标是 netlist 字段，不是标量路径。

\---

## 5\. 证据 schema（SQLite：`runs/evidence/evidence.db`）

```sql
CREATE TABLE ablation (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            TEXT,
  param         TEXT NOT NULL,        -- 搜索参数名
  layer         TEXT NOT NULL,        -- system/optical_core/topology/device
  workload      TEXT NOT NULL,        -- vit_tiny/vit_small/vit_base/mobilevit
  workload_features TEXT NOT NULL,    -- JSON：层数/MACs/计算vs存储瓶颈等（§7）
  old_value     TEXT,
  new_value     TEXT,
  hardware_map  TEXT NOT NULL,        -- JSON：DesignSpace.map 路径
  sub_arch_map  TEXT NOT NULL,        -- JSON：SUB_ARCH_PARAM_MAP 路径
  constraint_result TEXT NOT NULL,    -- pass / clamped / violated
  clamp_before  TEXT,                 -- 被夹前的原始值
  injection_confirmed INTEGER NOT NULL,
  injection_readback TEXT,            -- JSON：{path: 回读值}
  baseline_energy_pj REAL, baseline_area_um2 REAL,
  baseline_latency_s REAL, baseline_objective REAL,
  after_energy_pj REAL, after_area_um2 REAL,
  after_latency_s REAL, after_objective REAL,
  memory_energy_delta_pct REAL,       -- 不在 objective，单独记录
  metric_deltas  TEXT NOT NULL,       -- JSON：{energy_pct,area_pct,latency_pct,objective_pct}
  affected_metrics TEXT NOT NULL,     -- JSON 数组，如 ["latency_s","objective"]
  is_null        INTEGER NOT NULL,
  trustworthy    INTEGER NOT NULL     -- = injection_confirmed \&\& 单变量 \&\& deterministic
);
CREATE INDEX idx_param ON ablation(param);
CREATE INDEX idx_workload ON ablation(workload);
CREATE INDEX idx_param_workload ON ablation(param, workload);
```

\---

## 6\. 工程文件新增/修改清单

### 新增

|文件|职责|关键接口|
|-|-|-|
|`photon_tuner/evidence/__init__.py`|包声明|—|
|`photon_tuner/evidence/sub_arch_map.py`|第 3 层映射表 `SUB_ARCH_PARAM_MAP`|常量 dict|
|`photon_tuner/evidence/store.py`|证据持久化|`EvidenceStore.record/query/export`|
|`photon_tuner/evidence/probe.py`|消融探针（harness 核心）|`SearchSpaceProbe.probe_param/probe_all`、`_inject_and_readback`、`_evaluate`|
|`photon_tuner/evidence/workload_features.py`|负载特征提取（确定性，非 LLM）|`workload_features(model, adapter) -> dict`|
|`scripts/experiments/verify_search_space.py`|CLI 驱动|`--params --workloads --output-root --k`|
|`tests/test_evidence.py`|单元测试|覆盖 schema / 回读 / delta 计算 / null 判定|

### 修改（最小侵入）

|文件|改动|原因|
|-|-|-|
|`photon_tuner/simulators/netlist.py`|新增 `SUB_ARCH_PARAM_MAP`（或 import）|把隐式映射显式化，供回读|
|`photon_tuner/simulators/simphony.py`|新增 `read_sub_arch(simulator) -> dict` 辅助|暴露注入后的 `sub_arch_dictionary[name]` 供回读|
|`photon_tuner/core/workload.py`|若有现成 summary 则复用，否则加特征提取|供给 `workload_features`|
|（phase 2）`photon_tuner/rag/vector_store.py`|检索向量加入 workload 特征 + 已验证收益|RAG 集成（§8）|

> 不改 `build_sub_arch` 的写入逻辑本身（避免动到值保持的关键路径）；只加一张
> 并行映射表 + 断言测试，让它与 `build_sub_arch` 互相印证。

\---

## 7\. Workload 贯穿设计（特征提取）

每个负载在进入流程前，**一次性确定性提取**（不是 LLM，保证可信/可复现）：

```python
{
  "name": "vit_base",
  "layer_count": 61,
  "total_macs": ...,            # 总乘加数
  "operator_groups": {"projection": ..., "ffn": ..., "attention": ...},
  "memory_bytes": ...,          # 激活/权重内存足迹
  "bound": "compute" | "memory",  # 计算 vs 存储瓶颈判定
  "per_layer_profile_hash": ...  # 层剖面指纹（供 RAG 相似度）
}
```

该 dict 作为条件写进每条 `ablation` 记录的 `workload_features` 列，并贯穿
§3 的第 2/3/5/6/7/8 步。`bound`（compute/memory）直接决定「哪个参数预期会影响
哪个负载」——例如 `work_freq` 预期只影响 compute-bound 负载，`hbm_bandwidth` 预期
只影响 memory-bound 负载；验证结果与预期不符本身就是一条有价值的证据。

\---

## 8\. RAG 集成路径（phase 2，符合你「以后」的定位）

现在 `HistoryVectorStore` 是「75% 瓶颈语义 + 25% 硬件状态」双通道。证据库跑起来后，
把检索键升级为四通道：

```
75% 瓶颈语义 + 25% 硬件状态   （现状）
        ↓ 增加
+ workload 特征向量（bound/层数/MACs/剖面指纹）
+ 已验证收益向量（param × mapping × 指标 delta 方向/幅度）
```

这样 `solution_agent` 检索时，能匹配「在相似负载特征下、走同一映射、且已被证据
验证有收益的历史 case」，而不是只靠瓶颈文字描述碰运气。**这一步等证据库有数据、
人工核对可信后再做**（避免用未经核对的证据污染检索）。

\---

## 9\. 解读层：不新增智能体

**测量层必须是确定性 harness，而不是 LLM agent**——这套东西的价值就是「可信证据」，
一旦让 LLM 决定记什么、怎么判 delta，证据就失去确定性与可复现性。

**解读层也不再新增 agent。** 早期方案曾计划新增 `EvidenceAnalystAgent`
（`photon_tuner/agents/evidence.py`），该模块已删除：

* 它与已有的 architecture / topology / device 三个分析角色职责重叠——那三个角色本就
  在做「读证据 → 归因 → 给杠杆」；
* 多一个常驻角色会多一份提示词、多一轮 LLM 调用与一份 gate，却没有独立职责；
* 确定性证据已经在 `photon_tuner/evidence/`（`probe.py` 探测、`store.py` 落库、
  `sub_arch_map.py` 参数→sub_arch 路径映射）中产出，分析角色直接消费即可。

因此闭环简化为：

```
[离线] SearchSpaceProbe(确定性) → EvidenceStore(evidence.db)
          ├─ 喂 RAG（verified-gain 检索键，phase 2）
          └─ 喂下一次搜索 agent 的 prompt（verified-knowledge 段）
[在线] 下一次搜索：现有分析角色带着已验证知识 + RAG 跑
```

它仍然治已踩的坑：optical hallucination（SL_MZM「减衰减」被证据证伪）与 regressive
proposal（乱换器件导致 -853% 回归）——只是治法是**把确定性证据喂给已有角色**，
而不是再造一个角色。

\---

## 10\. 增量实施步骤与成本

|步骤|内容|成本|
|-|-|-|
|**P1 建机制**|`evidence/` 探测与落库 + `test_evidence.py`，跑通「P→V→注入→回读→仿真→delta→落库」对 1 个参数|纯代码，\~0 仿真|
|**P2 smoke**|`vit_tiny` × 3 个参数（work_freq / num_wavelength / 一个拓扑参数）各 2\~3 候选，核对证据链正确 + 回读真实|\~15 分钟（无 LLM）|
|**P3 放量**|代表性子集（强杠杆 + 从未触达参数证明 null）× 4 负载|\~1–2 小时|
|**P3.5 首轮解读**|把 `evidence.db` 的确定性 delta 喂给现有分析角色（不新增 agent），人工核对 verified_levers/null/anomalies|\~10 分钟（LLM）|
|**P4 全量**|17 架构 + 13 器件 × 4 负载 × 全候选|\~5–8 小时（可并行）|
|**P5 RAG**|四通道检索 + 证据回溯|待 P3/P4 数据|

> 当前扫描（17/20）与收敛实验（vit_base×5）仍在跑；P1 是纯代码不争资源，P2 的
> 15 分钟 smoke 可与它们错峰。

\---

## 11\. 与现有代码的复用关系

* 复用 `_inject_hardware`（`simphony.py`）+ `build_sub_arch`（`simulators/netlist.py`）做注入；
* 复用 `DesignSpace.violations/normalize/set`（`core/search_space.py`）做约束与映射；
* 复用 `materialize_hardware`/`step_arch_parameter`/`modify_arch_parameter`
（`core/search_space.py`）做单变量扰动；
* 复用 `tests.factory:build_<workload>` 与 `SimphonyAdapter` 做负载与仿真；
* **不经过 LLM/agent**——这是纯确定性 harness，证据可信且可复现。

\---

## 待你确认的 3 个点

1. **文件归属**：`SUB_ARCH_PARAM_MAP` 放 `evidence/sub_arch_map.py`（新增）还是直接
放 `simulators/netlist.py`（就近）？我倾向 `netlist.py` 就近、`evidence/` 只 import。
2. **P3 代表性子集的参数清单**：我建议
`work_freq, core_height, core_width, num_wavelength, dataflow, tiles, precision_in_bit, hbm_bandwidth, sharing_policy, laser_distribution` +
器件 `mzm, coupler, photodetector`（覆盖强杠杆 + 从未触达 + 拓扑 + 器件四类）。
3. **证据库位置**：`runs/evidence/evidence.db`（默认）还是独立 `data/evidence.db`？

