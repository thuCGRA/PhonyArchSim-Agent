# SimPhony 三层协同搜索系统:创新点与消融实验

本文回答两个问题:(1) 本项目的创新/贡献点是什么;(2) 如何用**对照消融实验**证明每一处贡献确实带来了可观测的增益。

消融实验全部**离线、确定性、可复现**(无真实 LLM 调用),证据由
`scripts/experiments/ablation_evidence.py` 产出,运行方式:

```bash
PYTHONPATH=. .venv-agent/bin/python scripts/experiments/ablation_evidence.py
```

---

## 0. 贡献点总览

| # | 贡献/创新点 | 对应完成标准 | 消融实验 | 关键证据(实测) |
|---|---|---|---|---|
| C1 | 联合协调 agent + 跨层共享约束(work_freq ≤ min 链、in_bit ≤ DAC、out_bit ≤ ADC) | (3) | A | 违反约束的设计点被 `DesignSpace.violations` 检出 3 处、门控行 5.5 判 `fail→re-execute` |
| C2 | 跨范式搜索(crossbar ↔ MZI mesh Reck/Clements)+ Pareto 归档 | (4) | B | 前沿从 1 点扩到 2 点;mesh 面积 −92% 但 IL +9.2 dB(权衡两端) |
| C3 | 确定性顺序/联合搜索调度(`search_strategy ∈ {joint, sequential}`) | (4) | C | joint 一轮原子落地三层;sequential 逐层落地、中间单层移动被 best-state 丢弃 |
| C4 | 8 行分级验收门控(结构合法性行 5.5) | (1)(5) | D | 移除行 5.5 后,非可求值模板推荐由 `re-execute` 变为 `accept`(放行) |
| C5 | 零编造出处纪律(DOI 可解析 / 未报告字段标注) | (5) | E | 41 变体 100% 覆盖:14 可追溯 + 23 逐器件标注 + 4 placeholder,无空 ref 裸奔 |
| C6 | 分析式 MZI mesh 求值器(不运行 vendored 真实引擎) | (4) | F | evaluable 模板从 1 扩到 3(crossbar/Reck/Clements);Butterfly 诚实保持 non-evaluable |

---

## C1. 联合协调 agent + 跨层共享约束

**贡献声明**:solution 从「单层器件改参」升级为**联合协调 agent**,一次提出跨「架构—拓扑电路—器件」三层一致的设计点,并由确定性约束 `work_freq ≤ min(MZM/DAC/ADC/PD 速率链)`、`in_bit ≤ DAC 精度`、`out_bit ≤ ADC 精度` 兜底,杜绝「白拿的 work_freq」这类物理不可行解。

**消融设计**:

- 基线(full):硬件点 `work_freq=5 GHz`(合法),经 `DesignSpace.violations()` + 门控行 5.5。
- 消融条件(移除约束):同一硬件把 `work_freq` 拉到 `25 GHz`,模拟「协调器不检查共享约束」。

**假设**:移除跨层约束后,物理不可行点会穿过门控;约束存在时被行 5.5 拦截。

**指标**:`violations()` 检出数、门控行 5.5 状态、`decide()` 决策。

**实测证据**(`ablation_evidence.py` → A):

```
链速率上限 min_work_freq_ceiling = 10.0 GHz
[基线 legal work_freq=5]  violations = 0  -> gate 5.5 pass
[消融 work_freq=25  ]  violations = 3  -> gate 5.5 fail(re-execute)
    work_freq 25 GHz exceeds dac sample_rate 14 GHz
    work_freq 25 GHz exceeds adc sample_rate 10 GHz
    work_freq 25 GHz exceeds photodetector bandwidth 20 GHz
```

**结论**:跨层共享约束把 3 处不可行违规(分别来自 DAC/ADC/PD 三个不同层级器件)一次性捕获并判 `re-execute`;移除后该点会作为「合法点」进入优化循环。约束是**跨层一致性**的确定性保障,而非仅靠 LLM 自觉。

---

## C2. 跨范式搜索 + Pareto 归档

**贡献声明**:搜索空间不止一个电路范式——`dptc_crossbar`(Hamerly 2019)与 MZI mesh(`mzi_mesh_reck`/`mzi_mesh_clements`,Reck 1994 / Clements 2016)在同一 Pareto 前沿上联合优化,由范式无关的 `ParetoArchive` 归档非支配点。

**消融设计**:

- 基线(full):`{crossbar 8×8} ∪ {Clements N=8, Reck N=8}` 三范式入归档。
- 消融条件(单范式):仅 `{crossbar 8×8}` 入归档(等价于「无 mesh 求值器」的历史状态)。

**假设**:单范式前沿是跨范式前沿的严格子集;mesh 范式新增「面积↔插损」的另一端权衡点。

**指标**:前沿大小、各端点的 area / energy / IL。

**实测证据**(`ablation_evidence.py` → B):

| paradigm | area_um2 | energy_pj | IL(dB) |
|---|---|---|---|
| crossbar 8×8 | 26,286,040 | 753,921 | 6.12 |
| clements_mesh N=8 | 1,987,808 | 756,353 | 15.28 |
| reck_mesh N=8 | 1,987,808 | 787,026 | 23.58 |

```
[单范式 crossbar-only]  frontier size = 1  (kind = {dptc_crossbar})
[跨范式 crossbar+mesh]  frontier size = 2  (kind = {dptc_crossbar, mzi_mesh_clements})
```

**结论**:跨范式把前沿从 **1 点扩到 2 点**——Clements mesh 面积仅为 crossbar 的 **7.6%(−92.4%)**,代价是插损 +9.16 dB。没有跨范式搜索,面积敏感场景下的 mesh 方案根本不会出现在候选里。附带一条物理正确的事实:Reck(最坏路径深度 2N−3=13)IL 23.58 dB > Clements(深度 N=8)15.28 dB,同 MZI 数下被 Clements 支配——正对应 Clements 2016 相对 Reck 1994 的改进动机。

---

## C3. 确定性顺序/联合搜索调度

**贡献声明**:协同搜索的**层调度由宿主(Python)确定性强制**,而非交给 LLM 决定 agent 数量/顺序。`search_strategy ∈ {"joint", "sequential"}`:`joint` 每轮原子应用三层 edits;`sequential` 按 `architecture → topology → device` 每轮只落一层、其余记为 `deferred_edits`。两种模式可消融对比。

**消融设计**:

- 基线(joint):`max_iterations=1`,脚本化 solution 提出三层混合编辑(cores_per_tile、core_kind/node_kind、on_chip_laser)。
- 消融条件(sequential):`max_iterations=3`,同一模型、同一编辑,逐层落地。

**假设**:joint 一轮原子落地;sequential 逐层落地,且由于 best-state 保留,「只有与其它层组合才降目标的单层移动」会被中间态丢弃。

**指标**:每轮 `applied_actions` / `applied_link_edits` / `deferred_edits` / 落盘后的 `core_kind`、`cores_per_tile`、`on_chip_laser`。

**实测证据**(`ablation_evidence.py` → C):

```
strategy    iter focus         applied  link deferred  core_kind            cores laser
joint       1    joint         1        1    0         mzi_mesh_clements    1     LT_Laser
sequential  1    architecture  1        0    2         dptc_crossbar        1     Customized_Laser
sequential  2    topology      1        0    1         mzi_mesh_clements    1     Customized_Laser
sequential  3    device        0        1    1         dptc_crossbar        1     LT_Laser
```

**结论**:

- `joint` 一轮把「少核 + 切 mesh + 换低损激光」**原子落地**并一起求值。
- `sequential` 严格逐层:第 1 轮只落架构、第 2 轮只落拓扑、第 3 轮只落器件。
- **可观测的差异**:sequential 第 2 轮的 mesh 拓扑切换在 `state_strategy="best"` 下**没有保留到第 3 轮**(第 3 轮起点回退到 `dptc_crossbar`),因为「单独切 mesh」这一单层移动没有立即改善标量目标,而它与「换低损激光」的组合收益只有 joint 的原子应用才能捕获。这正是**联合协调**相对**贪心逐层**的可证明优势。

---

## C4. 8 行分级验收门控(结构合法性行 5.5)

**贡献声明**:每个 stage 的产出都要过 8 行分级门控;其中**行 5.5(结构合法性)**是确定性代码(非 LLM),校验 `recommended_template` 是否合法且**可求值**、以及 hardware 点是否满足耦合矩阵 + 数值跨层约束。

**消融设计**:

- 基线(full):topology 推荐 `mzi_mesh_butterfly`(已声明但 `evaluable=false`),过 `structure_legality_check` + `decide`。
- 消融条件(移除行 5.5):`decide([])`——同一推荐不再经过行 5.5。

**假设**:行 5.5 把「会产出不可求值设计点」的模板推荐拦截为 `re-execute`;移除后被放行。

**指标**:行 5.5 状态、`decide()` 决策。

**实测证据**(`ablation_evidence.py` → D):

```
row 5.5 status = fail  detail = recommended_template 'mzi_mesh_butterfly' is declared but not evaluable
[有行 5.5] decide -> re-execute
[无行 5.5] decide -> accept   (非可求值模板被放行)
```

**结论**:行 5.5 是可靠性的一道**确定性闸门**。移除它,一个会让仿真器无法运行的模板推荐会被 `accept`,污染优化闭环;保留它则稳定 `re-execute`。该行**不依赖 LLM 判断**,因此不可被「幻觉」绕过。

---

## C5. 零编造出处纪律

**贡献声明**:每个 cfgs 数字都要能回溯到真实论文(DOI 可解析);查不到的字段**明确省略或标注**而非编造。器件库 41 个变体分三类治理:`traceable`(14,带可解析 ref)/ `baseline`(23,逐器件 honest 标注,来源 arXiv:2411.13715)/ `placeholder`(4,明确排除出搜索)。错误引用(Butterfly 的 Tang 2021)经核验后**撤回**,不因「凑数」而保留。

**消融设计**(结构性消融,非运行时开关):对比「有无出处纪律」下的可追溯覆盖率。

- 基线(full):`audit_device_provenance()` 全量审计。
- 消融条件(移除出处层):baseline/placeholder 的 `ref` 为空且无 `note` → 归为「不可追溯」。

**假设**:出处纪律使可追溯/已标注覆盖率达到 100%;移除后存在「空 ref 裸奔」变体。

**指标**:`traceable` / `baseline` / `placeholder` 计数、覆盖率。

**实测证据**(`ablation_evidence.py` → E):

```
器件库变体总数 = 41
  traceable (DOI/URL 可解析) = 14
  baseline-annotated (逐器件标注) = 23
  placeholder (明确排除/标注) = 4
覆盖率 = 100%(无空 ref 裸奔)
```

**结论**:41 变体 100% 覆盖——没有一个数字「来源不明」。14 个 traceable 的 DOI/URL 均在前文经 Crossref 逐条解析通过(13 Crossref + 4 URL,0 失败);23 个 baseline 有 `BASELINE_PROVENANCE_NOTES` 逐器件说明;4 个 placeholder 明确排除出搜索空间。这是「可靠、真实」目标的结构性底座。

---

## C6. 分析式 MZI mesh 求值器

**贡献声明**:不运行 vendored 的真实 SimPhony 引擎(只读引用),用纯函数 `mesh_model.py`(Reck/Clements/Butterfly 的 MZI 数、深度、器件倍数——全部来自论文结构事实)+ `build_mesh_sub_arch` 接入 analytic 适配器,让 mesh 范式**可求值**。

**消融设计**:

- 基线(full):`evaluable_template_names()` 当前值。
- 消融条件(移除 mesh 求值器):仅 `dptc_crossbar` 可求值(mesh 点全部 `evaluable=false`,即 phase-2 之前的史前状态)。

**假设**:mesh 求值器把可求值范式从 1 扩到 3。

**指标**:`legal_template_names()` vs `evaluable_template_names()`。

**实测证据**(`ablation_evidence.py` → F):

```
legal templates      = [dptc_crossbar, mzi_mesh_butterfly, mzi_mesh_clements, mzi_mesh_reck]
evaluable templates  = [dptc_crossbar, mzi_mesh_clements, mzi_mesh_reck]
Butterfly evaluable = False(公式可追溯 Cooley-Tukey,光子出处待定,诚实保持 non-evaluable)
```

**结论**:mesh 求值器让 4 个已声明模板中的 3 个可求值(crossbar + Reck + Clements)。Butterfly 因光子实现出处无可靠 DOI 而**诚实保持 non-evaluable**——这本身就是 C5「零编造」纪律的体现:不为了「模板数好看」而编一个引用。

---

## 附:如何复现

1. 证据脚本:`scripts/experiments/ablation_evidence.py`(上述 A–F 全部证据)。
2. 回归测试(锁定上述行为,防退化):
   - A/C1:`tests/deep/test_objective_alignment.py`、`tests/test_design_space.py`
   - B/C6:`tests/test_mesh_eval.py`(含 `test_reck_is_evaluable_and_dominated_by_clements`)
   - C/C3:`tests/deep/test_sequential_joint.py`(10 例)、`tests/deep/test_mesh_end_to_end.py`
   - D/C4:`tests/deep/test_evaluate.py`、`tests/deep/test_topology_space.py`
   - E/C5:`tests/test_provenance.py`(含 `test_every_baseline_device_has_explicit_note`)

当前全量:`deep 181 OK` + `plain 131 OK`。
