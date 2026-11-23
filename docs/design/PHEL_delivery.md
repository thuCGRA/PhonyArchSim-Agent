# PHEL-Agent 本轮改动与复现手册

本文件记录一次成体系的工程改造：**修了什么、为什么、如何验证、如何复现两组对比实验**。
面向两个用途：给论文写作提供可直接引用的机制说明；给后续维护者一份不会踩同一批坑的清单。

---

## 1. 修复：架构层/拓扑层的搜索被"看不见"地屏蔽

### 现象

`runs/arch_search_mobilevit/REPORT_final.md` 的结论是"架构层 4 个旋钮已被挖尽，
各轮反复在同一处震荡"。

### 根因

`searchable_parameters` 工具只枚举了 `topology_space.PARAM_SCOPE` 里的
**5 个**参数（`core_height` / `core_width` / `num_wavelength` / `dataflow` /
`mesh_rank`），而设计空间实际有 **22 个**。因此：

* `tiles`、`cores_per_tile`、精度、HBM、三级缓存、`work_freq`、`sharing_policy`、
  `temporal_accum`、`laser_distribution`、`core_kind` 从未出现在"可搜索"清单里；
* 默认拓扑 `dptc_crossbar` 下 `mesh_rank` 是 `not_applicable`，于是**拓扑层的可搜索
  轴为 0**。

智能体按工具的答复行事，所以它"正确地"只用了 4 根旋钮——这不是模型能力问题，是
**工具面撒谎**。这类退化最难查：所有环节都自洽，只有搜索空间悄悄变小。

### 修复

`DesignSpace.scope_table(core_kind)` 以设计空间本身为准枚举全部参数，`PARAM_SCOPE`
只用于覆盖已声明的例外。`searchable_parameters` 改为消费该表，并新增
`searchable_by_layer` 按层分组。

### 验证

`tests/test_param_scope.py`、`tests/test_scheduler.py` 覆盖；另有一次端到端单变量探针
（`docs/design/search_space_verification.md` 口径）确认 **22 个参数全部能改变目标值**。

---

## 2. 扩大的架构搜索空间（含一个必须尊重的后端边界）

架构层的几何 / 内存 / 频率轴改为 2 的幂，上限至少到 64：

| 参数 | 候选 |
|---|---|
| `tiles` | 1, 2, 4, 8, 16, 32, 64 |
| `cores_per_tile` | 1, 2, 4, 8, 16, 32, 64 |
| `core_height` / `core_width` | 8, 16, 32, 64 |
| `num_wavelength` | 8, 16, 32, 64 |
| `work_freq` (GHz) | 2, 4, 8, 16, 32, 64 |
| `hbm_bandwidth` | 512, 1024, 2048, 4096 |
| `glb1/glb2/rf_cache_size` | 512, 1024, 2048, 4096 |

**两条必须尊重的边界**：

1. **缓存下界是 512 KB。** CACTI 在当前的 `block_size` / associativity 下对
   64/128/256 KB 直接报 `Invalid memory configuration provided for GLB1`。
   这是后端建模边界，把数字加回候选表只会让每次求值抛错。
2. **精度轴保持 2/4/8 bit。** 上界由 DAC/ADC 的 `prec` 与采样率链物理给定。

基线初始点同时对齐到网格：`16×16×16 @ 4 GHz / 512 KB`（原 `12×12×12 @ 5 GHz / 485 KB`
在网格外，会被 `normalize` 折返）。`configs/initial_hardware.json` 现在是**唯一**的数值
基线来源——`default_hardware()` 先构造结构骨架再用该 JSON 深度覆盖，消除两处各自硬编码
12×12×12 的重复源。

---

## 2b. 自洽性审计：四处"接口处不自洽"（全部已修）

项目表面上每个环节都自洽，但跨模块边界有四处口径不一致。它们都无法被单模块测试
发现，因此新增了可重复的审计脚本 `scripts/analysis/audit_consistency.py`。

### 2b.1 目标函数忽略存储能耗（75–90%）

`simu_energy` 只算计算能耗，`simu_latency` 却含存储延时。于是"计时算存储、计价不算
存储"。实测存储占总能耗 75–90%（vit_tiny 93%），且**改用总能耗后最优设计点会变**
（面积 113 mm² 的 iter_003 让位给面积 43 mm² 的 iter_005）。

修法：`energy_mj` = 计算 + 存储，分量暴露为 `energy_compute_mj` / `energy_memory_mj`。
目标量级放大约 5–14 倍（取决于工作负载），这是修正而非退化。

### 2b.2 真实模式面积被多算 43%

厂商 `simu_area` 的过滤条件写成 `if "node" not in v`，判断的是**字典内部是否含
`node` 键**，而不是键名。结果节点级器件被计入逐器件求和后，节点边界框又被加一遍：
111.12 vs 77.75 mm²。解析镜像的算法是对的，所以离线测试从未暴露它。

修法：聚合层按 `device_area + node_bbox` 重算，原始值留在 `area.vendor_total_um2`，
构成明细在 `area.composition`；`by_chip_type` 改为自建分组，保证加总等于 total。

### 2b.3 逐层存储能耗恒为 0

`memory.energy` 是 `{sub_arch: {HBM|GLB2|GLB1|RF: pJ}}`（按存储层级组织），而
`_rank_layers` 用**层名**去查它，永远查不到，于是逐层 `memory_energy_mj` 恒为 0——
智能体会据此认为存储能耗为零，而它其实是能耗主体。

修法：删掉这个伪字段，改为在载荷里给出 `energy_scope`（compute/memory/total +
`memory_by_level_mj`），并显式说明"逐层能耗只覆盖计算"。

### 2b.4 RAG 先例没有结局标记

提交给历史库的既有成功提案也有失败提案。只按描述相似度检索，智能体会把"上次这么改
是错的"读成"上次这么改是对的"；调度器的 ρ 也会把"检索到相似描述"误当成"有可用先验"。

修法：案例摘要新增 `outcome = {label: improved|worse|unchanged, delta_pct}`，
且 ρ 只把 `improved` / `unchanged` 的 top-1 案例算作"有先验"。失败案例仍进载荷（供避坑），
但不再抬高 ρ。

### 2b.5 证据年龄恒被算成 1

`_record_evidence` 存输出时没有带上来源轮次，于是 `evidence_age` 恒为 1、
`evidence_iteration` 恒为 `None`。流水线模式对外声称"容忍 1–2 轮证据陈旧"，
但实际可能用了更旧的诊断而不自知——这会让"陈旧证据下的归因"这一实验变量失真。

修法：证据记录携带 `_iteration`，年龄按真实轮差计算。回归测试要求观察到 `age ≥ 2`
（若年龄被写死为 1，该断言失败）。

### 2b.6 解析镜像的 fJ/MAC 与能量口径不同源

解析镜像的 `report.metrics` 里 `energy_mj` 已含存储能耗，但 `fJ_per_mac` 仍按计算
能耗算——同一个块内部描述了两个不同的能耗范围。审计脚本的"report.metrics 与
metrics.json 一致"检查抓到了它。

### 2b.7 候选智能体看不到"最近改了什么、结果如何"

诊断真实失败模式时发现：候选载荷里只有当前点，没有历史。于是它持续提出跨 5–6 个
参数的大幅改动（例如同时改 cores_per_tile、tiles、core 几何、精度与缓存），而这
个乘性目标里面积是因子，一次 2 倍面积换来 20% 能耗下降是净亏。传统基线只做单参数
原子移动，天然不犯这个错。

修法：候选载荷新增 `recent_attempts`（最近 3 轮的 objective / Δ% / 面积 / 编辑），
让"步长过大"成为可见事实。

### 2b.8 顺带发现（非缺陷，但需说明）

`laser_distribution` 在网格/权重阵列族下 `H*W` 与 `H*W*N` 数值相同，因为这两族的
`num_wavelength` 物理上被强制为 1。该轴在这些族下实际只有 2 个不同取值，不是 3 个。

---

## 3. 指标单位

全部换算到光电系统惯例量纲（`photon_tuner/core/units.py`，唯一换算点）：

| 规范键 | 单位 |
|---|---|
| `latency_ms` | ms |
| `energy_mj` | mJ |
| `area_mm2` | mm² |
| `fJ_per_mac` | fJ/MAC |
| `area_mm2_per_mac_per_ms` | mm²/(MAC/ms) |

关键性质：三项换算因子之积为 1，**目标函数数值在换算前后完全相同**（由
`tests/test_units.py::test_objective_is_invariant_under_conversion` 钉住）。因此历史
run、Pareto 前沿与已发表的对比数值全部保持可比，不存在"换了单位所以变好了"的歧义。

报告里保留 `si_metrics` 与 `units` 块，使规范值与 SimPhony 原生量可互相核对。详见
[units.md](units.md)。

---

## 4. 角色更名

| 旧 | 新 | 理由 |
|---|---|---|
| `model` agent | `bottleneck` agent | 其职责是**全局瓶颈提名**；"model" 与 LLM 模型、NN 模型三重混淆 |
| `solution` agent | `candidate` agent | 它产出的是**一个一致的候选设计点**，不是"求解" |

新链路：`bottleneck → architecture → topology → device → candidate → commit`。
提示词、载荷、工具白名单、图节点、落盘文件名、测试全部同步。

---

## 5. RAG 驱动的动态调度

新增 `photon_tuner/runtime/scheduler.py`：以 ρ（RAG 就绪度）、σ（瓶颈集中度）、
π（改进斜率）、κ（门返工率）、β（剩余预算）为输入的**确定性纯函数**，输出模式、
聚焦层与逐层证据偏移。

为什么不让 LLM 决定模式：调度输出必须是可复现的实验条件，否则"自适应 vs 固定模式"
的消融无法成立。分工是**决策在模型、调度与判定在宿主**。

`--schedule-strategy fixed` 是默认（等价历史行为，也是消融对照）；
`--schedule-strategy rag` 是论文正式设置。设计、优先级与消融协议见
[adaptive_scheduling.md](adaptive_scheduling.md)。

---

## 6. 两个会让长跑实验整批作废的坑（已修）

### 6.1 MCP stdio 被仿真器输出污染

`CACTI simulation completed...`、`[W] Cannot import torchsummary` 等普通文本会写进
MCP 的 stdio JSON-RPC 通道，客户端因此间歇性报
`JSONDecodeError: Expecting value: line 1 column 1` 而整轮实验中断。

根因不只是"没重定向"，而是**三个通道共用了同一条流**。修复后彻底分离：

* 协议 → 私有 fd（`os.dup(1)` 复制后在 `_serve()` 中显式传给
  `mcp.server.stdio.stdio_server(stdout=...)`）；
* fd 1 → `/dev/null`（吃掉 C 层、子进程、CACTI 的原生输出）；
* `sys.stdout` → `/dev/null`（吃掉模拟器的 Python `print`）。

隔离必须发生在**任何重型 import 之前**，否则导入期警告已经污染了管道。

客户端同时改为在解析失败时把收到的原始文本带进错误信息——没有这条，现场只有一句
`Expecting value`，无法定位。

### 6.2 MCP 模式下工作负载被静默忽略

`SimphonyAdapter(nn_model_factory=...)` 在 `transport="mcp"` 下**不生效**：工作负载由
服务端环境变量 `SIMPHONY_NN_FACTORY` / `SIMPHONY_WORKLOAD` 决定，缺省回退到 `vit_tiny`。
CLI 与基线 runner 都已正确设置这些变量，但直接构造适配器时很容易忘——忘掉的症状是
"mobilevit 跑出 vit_tiny 的数"，而**面积恰好一致**（面积只取决于硬件），极难察觉。

### 6.3 仿真器拒绝的提案不再拖垮整轮

真实 SimPhony 会对物理/后端不可行的点直接抛错（例如上述 CACTI 缓存配置）。现在的语义是：

* 基线：`EvalRecord.metrics = {"objective": inf, "simulation_error": ...}`，搜索继续；
* agent：写 `simulation_rejected.json`，**保持当前点不变**，调度历史记 `accepted=false`。

设计空间本来就是非平滑、带洞的；一次坏提案不应该烧掉整轮预算。

---

## 7. 复现两组对比实验

```bash
# 一次性跑完：5 个传统基线 + agent，10 次提案 / 1 seed
export OPENAI_API_KEY=...            # DeepSeek 官方 key
.venv-agent/bin/python scripts/experiments/phel_experiments.py all \
    --workload mobilevit --iterations 10 --seeds 0 \
    --schedule-strategy rag \
    --llm-provider deepseek-official --llm-model deepseek-chat \
    --llm-base-url https://api.deepseek.com/v1
```

> `--llm-base-url` 必须显式给：`configs/llm.json` 的 `base_url` 优先级**高于** provider
> 预设，不给就会把官方 key 发到代理端点并得到 401。

产物布局（每组实验一个文件夹，两个文件 + 一份可读摘要）：

```
runs/exp1_equal_budget/
    experiment_data.json     # 数据：逐方法改进幅度、收敛曲线、逐 run 明细、公平性契约
    llm_run_records.json     # 大模型运行记录：逐次调用的 tokens / 耗时 / 推理 token
    summary.md               # 由数据自动渲染的表格（避免手抄数字）
runs/exp2_convergence/
    experiment_data.json     # 同上 + evals-to-target 与相对智能体的节省百分比
    llm_run_records.json
    summary.md
```

实验二默认**复用**实验一的运行产物（同一批 run 换一种统计口径），不重复烧算力；
`--recompute-convergence` 可强制单独跑一轮。

两组实验的公平性契约（`experiment_data.json: fairness_contract`）：

* 同一引擎：SimPhony real（`ONNArchSimulator`，经 MCP）；
* 同一搜索空间：`configs/search_space.json` + `configs/initial_hardware.json`；
* 同一目标：`latency_ms × fJ_per_mac × area_mm2`（含 IL 预算软惩罚）；
* 同一动作空间与 keep-best 规则；
* 预算横轴统一为 **SimPhony 求值次数**：agent `--iterations N` 与基线 `--budget N`
  都是 1 次初始求值 + N 次提案。

---

## 8. 测试

```bash
.venv-agent/bin/python tests/run_tests.py     # 411 项，全绿
```

新增测试：

| 文件 | 锁住的性质 |
|---|---|
| `tests/test_scheduler.py` | 调度纯函数的判定优先级、聚焦层轮转、偏移计划、确定性 |
| `tests/deep/test_schedule_integration.py` | 决策真的传导到阶段跳过、证据偏移与落盘产物 |
| `tests/test_units.py` | 换算正确性 + **目标函数在换算下不变** |
