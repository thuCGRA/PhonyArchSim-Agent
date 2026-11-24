# 可复用工具箱、原子技能与组合工作流

## 1 三层能力边界

本项目将 Agent 可使用的能力拆为三个互不混淆的层次。

1. **工具（Tool）**：代码能力或外部系统接口，执行确定性操作，不包含 Agent 的专业判断策略。工具只声明输入、输出、资源分类、访问级别与错误语义。
2. **原子技能（Atomic Skill）**：Agent 可独立理解和调用的最小完整专业任务。一个技能可以固定调用一个或多个工具，但只回答一个专业问题；它声明前置条件、所需证据、工具计划、判断规则、结果契约、失败策略和执行轨迹。原子技能不调用其他技能，也不含任务级分支或循环。
3. **组合工作流（Composite Workflow）**：多个原子技能的有向图，负责条件分支、循环、重试、早停与阶段衔接。

形式上，工具记为 `T=(name, resource, access, I, O)`；原子技能记为
`S=(goal, preconditions, evidence, tool_plan, rule, O, failure, trace)`；组合工作流记为
`W=(skills, edges, conditions, termination)`。这一划分使“确定性执行”和“专业判断”在代码与论文中均可独立审计。

## 2 工具箱：按被操作资源分类

`photon_tuner.toolbox.catalog` 统一登记 40 个工具。分类依据是工具操作的资源，而不是调用它的 Agent 角色，因而同一工具可被多个角色复用。

| 工具箱分类 | 数量 | 功能边界 |
| --- | ---: | --- |
| reports | 14 | 读取硬件、指标、能耗、面积、损耗、层/算子和负载报告 |
| design_space | 2 | 查询设计空间与当前生效参数 |
| topology | 5 | 查询拓扑模板、光路和光学结构 |
| devices | 4 | 查询器件配置、候选、前沿和来源 |
| simulation | 2 | 预览评估契约或由宿主执行一次仿真 |
| state_memory | 8 | 读取搜索状态、设计身份、案例记忆、Pareto 前沿和调度信号，或提交已核验案例 |
| validation | 1 | 确定性检查跨层耦合约束 |
| editing | 4 | 由宿主修改架构、拓扑和器件参数 |

访问控制进一步分为 33 个只读工具、1 个分析上下文控制工具和 6 个写工具。34 个非写工具可进入 Agent 白名单；`modify_*`、`step_hardware_parameter`、`evaluate_design` 和 `commit_verified_case` 等 6 个写工具只允许确定性宿主调用。这样 Agent 负责诊断和建议，状态变更、昂贵仿真和记忆提交仍受宿主验证与审计。

## 3 原子技能：按专业问题分类

`photon_tuner.skills` 注册 15 个原子技能。技能名全部采用“判断/识别/选择”语义，与读取或修改资源的工具名明确区分。

| 技能分类 | 数量 | 原子技能 | 回答的专业问题 |
| --- | ---: | --- | --- |
| diagnosis | 6 | `assess_memory_energy_dominance`、`identify_latency_hotspot`、`identify_dominant_energy_device`、`identify_dominant_area_device`、`assess_optical_power_budget`、`identify_dominant_optical_loss` | 当前瓶颈、主导器件或光链路余量是什么 |
| decision | 3 | `choose_analysis_baseline`、`assess_objective_improvement`、`select_device_frontier_candidate` | 应比较哪个基点、候选是否改善、器件前沿上选哪个点 |
| verification | 3 | `assess_parameter_effectiveness`、`assess_candidate_legality`、`assess_design_novelty` | 参数是否生效、候选是否合法且未重复 |
| learning | 2 | `assess_case_reusability`、`assess_memory_maturity` | 历史案例能否复用、案例记忆是否成熟 |
| scheduling | 1 | `assess_objective_convergence` | 近期目标是否已经收敛 |

例如，`summarize_topology_evidence` 只是返回损耗、器件和功率证据，因此是工具；`assess_optical_power_budget` 在证据上应用链路余量规则并回答“链路是否可行”，因此是原子技能。后者会记录所用工具、实参和原始输出，形成可追溯执行轨迹。

## 4 组合工作流：负责控制流

`photon_tuner.workflows` 当前实现 3 个可复用组合工作流：

| 工作流 | 技能组成 | 分支语义 |
| --- | --- | --- |
| `candidate_preflight` | 合法性判断 → 新颖性判断 | 只有候选合法时才检查重复，避免无效仿真 |
| `optical_path_diagnosis` | 光功率预算 → 主导损耗识别 | 联合形成链路可行性与根因诊断 |
| `memory_aware_search_review` | 记忆成熟度 → 目标收敛判断 | 只有先验足够时才依据收敛状态切换策略 |

工作流运行器以条件边和最大步数执行技能图；技能自身仍保持无分支、单一目标。后续可在工作流层加入修复循环、仿真后复核和失败回退，而不改变工具或原子技能契约。

## 5 Agent 能力面与运行时集成

六类 Agent 通过显式白名单选择工具和技能：瓶颈、架构、拓扑、器件、候选与评估角色分别拥有 20、23、18、11、18、45 项能力。白名单中的技能由 `build_atomic_skill_tools()` 包装为 Agent 可调用接口；技能运行时再调用已绑定的确定性工具。编排器在每轮开始时绑定当前设计、报告、案例库、Pareto 档案、调度信号和仿真适配器。

`validate_agent_capability_surface()` 自动检查三项不变量：所有白名单名称均已注册、角色内不存在重复项、任何角色都不能直接调用宿主写工具。`scripts/analysis/capability_inventory.py` 会从代码生成 JSON 和 Markdown 清单，供实现审计和论文制表共用。

## 6 可扩展机制

新增能力时遵循以下规则：

1. 若只是新的确定性接口，在对应工具箱实现并加入统一目录；不强行包装成技能。
2. 若能回答一个新的专业问题，声明一个原子技能，固定其证据工具和判断规则；不得在技能内调用其他技能。
3. 若任务涉及技能间条件、循环或返工，在组合工作流中添加节点和边。
4. 将只读工具或技能加入必要角色的最小白名单；写工具继续由宿主掌控。
5. 运行清单覆盖、权限面、判断规则和条件分支测试，再重新生成能力清单。

由此，新工具、新专业判断和新搜索流程可以分别扩展，避免修改一个能力时牵连全部 Agent 和主流程。
