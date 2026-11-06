# 第 5 章 Evaluation — 设计与实验矩阵

本文是论文 §5（1 页 Evaluation）的设计稿。所有事实以当前代码与已有产物为准；
带 ⟨…⟩ 的是需要你填的实测值，带 ⚠️ 的是**现有数据不足以支撑、必须补跑**的项。

---

## 0. 先修正提纲中三处与代码不符的前提

| 你的提纲 | 实际情况 | 证据 |
|---|---|---|
| "实现基于 **DeepAgent** 框架" | **不是** DeepAgents 库。实现是原生 **LangChain + LangGraph**：`langgraph.prebuilt.create_react_agent` 构建 6 个智能体，`StateGraph` 编排父图，`SqliteSaver` 做检查点。`import deepagents` 会 `ModuleNotFoundError` | `photon_tuner/agents/builders.py:19`、`runtime/orchestrator.py:22,538` |
| "运行在 **XX 的电脑配置**" | 单机，**CPU only**：Intel Core Ultra 9 275HX，WSL2 分配 4 vCPU / 9.7 GiB，Ubuntu 26.04（内核 6.18.33.2-microsoft-standard-WSL2），无可用 GPU | `lscpu` / `free -h` / `uname -r` |
| "实验使用的几种 **XX 模型**" | 内置 3 个真实模型预设 + 1 个确定性假模型：`DeepSeek-V4-Pro-0813`、`DeepSeek-V4-Flash`、`deepseek-chat`、`mock` | `photon_tuner/llm/model.py:78-94` |

> 建议表述："The agent runtime is built on native LangChain + LangGraph
> (`create_react_agent` per role, a parent `StateGraph` for orchestration,
> `SqliteSaver` for resumable checkpointing), **not** on the DeepAgents library."
> 顺带把"顺序由宿主 Python 控制、不由模型决定"写进来——这是可复现性的关键设计。

---

## 1. §5.1 Setup

### 5.1.1 智能体实现与运行环境

写成一段 + 半张表：

- **运行时**：Python 3.14（≥3.11 因 langgraph 要求）
- **流水线**：`model → architecture → topology → device → solution → commit`，
  6 个已编译 LangGraph 智能体；每阶段以 Pydantic `structured_response` 收尾
- **可恢复性**：父图每节点后写检查点，崩溃可 `resume_run_dir` 续跑
- **硬件**：Core Ultra 9 275HX / 4 vCPU (WSL2) / 9.7 GiB / CPU-only / Ubuntu 26.04
- **CACTI**：本地编译，CPU 运行
- **为什么强调 CPU-only**：仿真与全部搜索都在 CPU 上，**LLM 通过远程 API**。
  所以 wall-clock 主要由「仿真次数」与「LLM 往返」决定，GPU 不参与。
  这句话能挡掉"你们是不是靠算力堆出来的"这类质疑。

### 5.1.2 仿真引擎与扩展（这是你最该展开讲的一节）

底引擎是 SimPhony（vendored 副本，arXiv:2411.13715），经 **MCP stdio 桥在
Python 3.10 进程外**运行（torch 1.12 栈与主运行时冲突）。在此基础上做了 7 处扩展，
**其中 E1、E3 是让实验能够成立的前提，必须写进正文**：

| # | 扩展 | 作用 | 若不做的后果 |
|---|---|---|---|
| **E1** | **硬件注入**（`_inject_hardware`）：从被搜索的 `hardware` 字典重建 sub-arch，替换仿真器解析结果 | 让搜索**真的生效** | 上游 `ONNArchSimulator` 从固定 YAML 读架构、**完全忽略**传入的硬件字典 → 每次求值都仿真同一个 LT 基线，**搜索是空转** |
| **E2** | 非 mesh 拓扑网表装配（3 类：crossbar / WDM 权重阵列 / PCM 交叉开关） | 把硬件参数展开成器件级网表 | 只有一种电路范式可评估 |
| **E3** | **解析式 MZI mesh 求值器**（Reck / Clements / Butterfly 几何模型） | 可评估模板 **1 → 3** | mesh 范式缺席，跨范式搜索无从谈起 |
| **E4** | **细粒度仿真报告**：`area_detail / energy_detail / insertion_loss / memory / report.json` 逐项外露并包装成智能体工具 | 智能体能看到"能量花在哪、IL 花在哪"，而不是只拿到一个标量 | 智能体无法做成本归因，退化成本文 §3 的盲目改参 |
| **E5** | 激光功率双键写入（上游 `laser_power_distribution` vs 读取 `laser_power_distribution_factor` 字段名不一致） | 让光功率预算真实生效 | 上游静默丢弃自身配置：能量 9.02e9 vs 修正后 9.81e9 pJ |
| **E6** | 器件出处审计（41 个变体的 DOI/标注覆盖）+ 片上存储能量多计修正（`/ block_size`） | 数字可追溯、不虚报 | 审稿人会质疑器件参数是编的 |
| **E7** | 跨层共享约束检查器（`work_freq ≤ min(MZM/DAC/ADC/PD)`、`in_bit ≤ DAC`、`out_bit ≤ ADC`） | 确定性拦截物理不可行点 | 见 §5.3 C1 |

> 建议在正文用一句总括："We extend SimPhony from a *single-paradigm, black-box
> scalar evaluator* into a *multi-paradigm, attribution-ready* simulator:
> hardware injection (otherwise the searched design is silently ignored),
> three non-mesh netlists, an analytic MZI-mesh evaluator, and per-component
> detail reports exposed as agent tools."
>
> E1 那条一定要单独强调一句——**"the upstream simulator ignores the hardware
> dict"** 是很有分量的工程发现，说明你的工作不是简单调参。

### 5.1.3 底层大模型

| preset | 模型 | 端点 | 用途 |
|---|---|---|---|
| `paratera` | DeepSeek-V4-Pro-0813 | llmapi.paratera.com | 主实验（§5.2/§5.3） |
| `paratera-flash` | DeepSeek-V4-Flash | llmapi.paratera.com | §5.4 模型无关性 |
| `deepseek-official` | deepseek-chat | api.deepseek.com | §5.4 模型无关性（**跨供应商**，最有说服力） |
| `mock` | `FakeToolCallingModel`（确定性） | 无网络 | 框架确定性对照，无 API 成本 |

要点：
- 密钥统一走 `OPENAI_API_KEY` 环境变量，不进任何配置文件（可写一句，体现工程规范）
- `mock` 的存在让你可以做**「框架确定性」**实验：同一输入跑 N 次，指标逐位相同。
  这是很有力的可复现性证据，建议在 §5.3 顺带报一句
- 跨供应商（paratera vs deepseek 官方）比同供应商两个型号更能支撑"与模型无关"

### 5.1.4 对比方法（解释每个方法 + 为什么公平）

| 方法 | 一句话说明 | 选它的理由 |
|---|---|---|
| **RS** 随机搜索 | 当前最优点的合法邻域内随机游走 | 最简黑盒下界 |
| **SA** 模拟退火 | log-objective 空间几何冷却（t₀=0.20, α=0.97） | 光子器件/电路级调优的经典手段 |
| **BO** 贝叶斯优化 | Matern(ν=5/2) GP + EI，对 −log10(obj) 拟合 | 电子加速器 DSE 的 SOTA 基线 |
| **GA** 遗传算法 | 分组交叉 + 逐基因变异 + 精英保留，pop=12 | 光子 NAS/MVM 处理器的主流路线 |
| **Sweep** 坐标扫描 | 确定性单参数穷举（补充基线） | 确定性、可复现的上限型参照 |

**必须先讲公平性契约，否则数字不可比**（这段是审稿人必查的）：

1. 同一评估器：全部走 `SimphonyAdapter` → 真实 `ONNArchSimulator`
2. 同一搜索空间：同一份 `initial_hardware.json` + `search_space.json`
3. **统一动作空间**：传统算法只能做「参数移一格 / 切一次模板 / 换一个器件变体」
   的最小合法编辑，与智能体工具箱能力对齐 → **差异只来自决策策略，不来自可达性**
4. 同一预算口径：横轴统一为 **SimPhony 求值次数**；`--iterations N` 与
   `--budget N` 都是 **1 次初始求值 + N 次提案**
5. 同一产物格式：基线 `trajectory.json` 是智能体产物的字段超集，同一解析器读取

> 第 3 条是整个对比成立的支点，务必写。不然"智能体动作空间更大所以赢了"就是硬伤。

### 5.1.5 Benchmark 表

**表 1：Benchmark 配置**（建议就放这一张）

| ID | Workload | 模型 | 量化层数 | 初始硬件 | 搜索维度 |
|---|---|---|---|---|---|
| B1 | vit_tiny | QBertViTBase (dim192, d4, h3) | 21 | 12×12, 4 tiles, 4-bit, f=5 GHz | arch 14 + topo 6 + device |
| B2 | vit_small | QBertViTBase (dim384, d6, h6) | 31 | 同上 | 同上 |
| B3 | vit_base | QBertViTBase (dim768, d12, h12) | 61 | 同上 | 同上 |
| B4 | mobilevit | QMobileViT (224×224) | 78 | 同上 | 同上 |

- **预算**：⟨N⟩ 次 SimPhony 求值（建议 25 或 50，两者都报最好）
- **重复**：每个 (workload, method) ⟨10⟩ 个种子/随机运行
- **初始硬件**：`configs/initial_hardware.json`（crossbar 起点）+ 另报一个 mesh 起点

理由（写进正文）：4 个 workload 覆盖 21→78 层、CNN/Transformer 两个映射
（`simple_cnn` / `simple_transformer`），能量跨 ~45×，能验证指标模型自洽性。

---

## 2. §5.2 与传统方法对比

### 2.1 指标（每个都要给定义）

| 指标 | 定义 | 为什么要有 |
|---|---|---|
| **Normalized objective** | best / initial（越低越好） | 主指标，跨 workload 可加总 |
| **Improvement %** | 1 − best/initial | 更直觉，适合正文叙述 |
| **Evals-to-target** | 达到 X% 改进所需求值数 | **样本效率**，最能体现智能体价值 |
| **Success rate** | 有正改进的运行比例 | 稳健性，防"平均分被个别好运行拉高" |
| **Wall-clock** | 每运行耗时 (s) | 效率（要注明 CPU-only、含 LLM 往返） |
| **LLM tokens / calls** | 每运行 token 与调用数 | **成本**，回答"多花的钱值不值" |

`tokens` 是可得的：`common/metrics.py` 聚合每次调用的 `token_usage` 写进 `metrics.json`；
`trace/prompt_calls.json` 有逐次调用的耗时与状态。

### 2.2 表 2：主对比表

| Workload | Method | Norm. obj ↓ | Impr. % ↑ | Evals | Success ↑ | Wall (s) | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| B1 | Agent (ours) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| B1 | RS / SA / BO / GA / Sweep | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | — |
| B2–B4 | 同构 | | | | | | |

- 每个数字写成 **mean ± std over ⟨10⟩ seeds**
- 传统方法 Token 列填 "—"（不耗 LLM），**这个空栏本身就是卖点**

### 2.3 图 2：收敛曲线

已有工具：`baselines/plotting.py` 出零依赖 SVG；`baselines.compare` 一键生成
`curves_*.svg` + `bars_*.svg`。
横轴 = 求值次数，纵轴 = normalized objective，每个 workload 一张，
智能体 vs 5 条基线。**同一预算截断**，这点要在图注里写死。

### 2.4 必须放进正文的三条分析

1. **样本效率**：智能体在 ⟨N⟩ 次求值内的改进 vs 基线需要 ⟨M⟩ 次才能达到 →
   若 M ≫ N，这是最有力的结论
2. **失败归因**：`scripts/analysis/analyze_sweep.py` 已能自动区分
   **LLM 驱动** vs **规则回退**（`_repair_duplicate` 的 work_freq 逃逸）。
   报告"纯 LLM 驱动占比"⟨x⟩%，避免把回退收益算成智能体能力
3. **回归防护**：`keep-best` 拒绝了多少次负改进提案（如 vit_base 上
   area +322.7% 的回归被拒）→ 说明"智能体不总是对的，但框架不退化"

---

## 3. §5.3 消融实验

分三层做，从粗到细。

### 3.1 表 3：系统组件消融（你的 C1–C6）

全部**离线、确定性、无 LLM**，证据由 `scripts/experiments/ablation_evidence.py`
的 A–F 六节直接产出（已有实测值，可直接引用）：

| # | 消融项 | Full | Ablated | 指标 | 实测证据 |
|---|---|---|---|---|---|
| C1 | 跨层共享约束 | 合法点 violations=0 → gate 5.5 `pass` | `work_freq=25` → **3 处违规** → `fail→re-execute` | 违规检出数、门控判定 | 链上限 10 GHz；DAC 14 / ADC 10 / PD 20 GHz 三处同时被捕获 |
| C2 | 跨范式搜索 | 前沿 **2** 点（crossbar + Clements） | 仅 crossbar → 前沿 **1** 点 | 前沿大小、两端权衡 | Clements 面积 **−92.4%**（1.99e6 vs 2.63e7 µm²），代价 IL **+9.16 dB** |
| C3 | 联合 vs 顺序调度 | joint：一轮原子落地三层 | sequential：逐层落地，其余 deferred | 落地层数/轮、deferred 数 | joint 1 轮 3 层全落地；sequential 3 轮逐层 |
| C4 | 门控行 5.5（结构合法性） | 非可求值模板 → `re-execute` | 移除该行 → **`accept`（放行）** | 决策翻转 | Butterfly 模板推荐被挡 |
| C5 | 出处纪律 | 41 变体 **100%** 覆盖 | 无标注裸奔 | 覆盖率 | 14 可追溯 + 23 逐器件标注 + 4 placeholder |
| C6 | 解析式 mesh 求值器 | 可评估模板 **3** | 收缩为 **1**（仅 crossbar） | 可评估模板数 | Reck/Clements 可评估；Butterfly 诚实标 non-evaluable |

> C2 有个很漂亮的附带结论可以写：Reck IL 23.58 dB > Clements 15.28 dB，
> 同为 N=8 时 **Reck 被 Clements 支配**——这正好对应 Clements(2016) 相对
> Reck(1994) 的改进动机。用仿真复现了文献里的已知结论，是很强的正确性论据。

### 3.2 表 4：运行期开关消融（CLI 已支持，能直接跑）

| 消融 | 开关 | 对照 |
|---|---|---|
| 证据门 | `--no-gate` | vs 默认（8 行协议） |
| 搜索调度 | `--search-strategy sequential` | vs `joint` |
| 跨范式预扫 | `--topology-sweep` 关 | vs 开 |
| 状态继承 | `--state-strategy next` | vs `best`（keep-best） |
| RAG 历史 | 空 store | vs 有历史案例 |
| LLM 能力 | `--provider mock` | vs 真实模型（分离"框架"与"模型"贡献） |

**这张表价值极高**：全部是脚手架里已有的开关，不用改代码就能跑出 6 组对照。

### 3.3 ⚠️ 角色消融（需要加一个开关）

"去掉 model / architecture / topology / device 中某一个智能体"是最标准的组件消融，
但**当前 CLI 没有对应开关**（`cli.py` 无 `--disable-stage`）。

选项：
- **A（推荐）**：加一个 `--disable-stage {model,architecture,topology,device}`，
  被禁用的阶段直接输出空结论（不调 LLM），5 组对照
- **B（省事）**：用 `--search-strategy sequential` 的层轮转间接论证"每层都有用"
- **C（最弱）**：只报 3.1 + 3.2，在 limitation 里说明

若走 A，改动量约 30 行（`cli.py` 加参数 + `orchestrator._run_*_stage` 提前返回），
我可以直接实现。

---

## 4. §5.4 不同大模型的结果

### 4.1 表 5：模型无关性

| Model | Vendor | Norm. obj ↓ | Impr. % ↑ | 纯 LLM 驱动率 | Tool-call 错误率 | Tokens/run | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Pro-0813 | paratera | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| DeepSeek-V4-Flash | paratera | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| deepseek-chat | DeepSeek 官方 | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| `mock`（确定性） | — | ⟨⟩ | ⟨⟩ | — | — | — | ⟨⟩ |

**表要传达的核心结论**：改进幅度随模型波动，但**方向一致**（都优于初始点、
都不如人工调参的上界），即"框架能力与模型无关，模型只影响效率与上限"。

### 4.2 必须单独讲的：失败模式分类

这一小节是全章最有说服力的地方——**坦率报告失败比只报涨幅更容易被信任**。
你的 `docs/results/results.md` 已经记录了真实失败，直接归纳成表：

| 失败类型 | 实例 | 框架如何兜住 |
|---|---|---|
| 工具参数名错 | 提 `in_bit` 而非 `precision_in_bit` | 校验失败 → `_repair_duplicate` 回退 |
| 幻觉器件替换 | 无人检查面积就换 `ACADEMIA_EO_MZM_1x1`（比 LT_MZM 大 141×） | 仿真真值让 objective 变差 → keep-best 拒绝 |
| 事实错误陈述 | 称 SL_MZM"降低衰减"，未查其 4.5 dB IL | 同上，净收益仍为正 |
| 回归提案 | vit_base 上 height/width 12→16 → area +322.7% | keep-best 拒绝，交付基线 |

论点：**"LLM 会犯错，但框架的确定性护栏（约束检查 + 门控 + keep-best）把
错误限制在'不进步'而非'变差'"**。这比声称"LLM 很准"稳得多。

### 4.3 建议加一句成本核算

`Pro` vs `Flash`：⟨Flash 省 X% token、慢 Y%、差 Z 个点⟩ → 给出"性价比"结论。
这对实际落地很有价值，且只有你有多模型数据能说。

---

## 5. 实验矩阵与预算

基于实测：**一次 1-iteration 的 vit_tiny 运行（2 次求值 + 5 个阶段 LLM）
耗时 390–690 s**（`runs/run_agent/legacy/sweep_buggy_latency/sweep_summary.jsonl`）。
即含 LLM 的求值约 **200–350 s/次**。基线无 LLM，单次求值更快（⟨实测⟩）。

| 实验 | 组合数 | 每组合求值 | 合计求值 | 估算机时 |
|---|---|---:|---:|---:|
| §5.2 智能体 | 4 workload × 10 seeds = 40 运行 | 1+25 | 1040 | ~50–90 h |
| §5.2 基线 | 4 × 5 方法 × 10 seeds = 200 运行 | 1+25 | 5200 | ⟨待测⟩ |
| §5.3 开关消融 | 6 组 × 4 seeds | 1+25 | 624 | ~15–20 h |
| §5.4 多模型 | 3 模型 × 4 workload × 3 seeds = 36 | 1+25 | 936 | ~30–50 h |

**结论：全做完需要数百小时单机时间。必须降预算。** 三个可行方案：

- **降预算**：25 → 10 次求值（曲线仍能体现样本效率），总机时降到 ~1/2.5
- **降种子**：10 → 5（statistical claim 变弱，但可在 limitation 说明）
- **降 workload**：§5.4 只用 B1+B3 两个 workload 代表小/大模型

建议组合：**主表 25 evals × 10 seeds（B1–B4）；消融 15 evals × 5 seeds**。
这样 §5.2 约 40–60 h，可接受。

---

## 6. ⚠️ 现有数据的缺口（必须先补跑）

**结论：当前仓库里的数字没有一个能直接进论文。** 具体缺口：

| 数据源 | 现状 | 问题 |
|---|---|---|
| `runs/run_baseline/manifest.json` | 50 条，5 算法 × 10 seeds × 41 次求值 | workload 字段是 **`synthetic`**，不是 B1–B4；不可用于主表 |
| `reports/agent_vs_baselines_b2/b3` | agent 2 条 / 基线 3 次求值 | **只是冒烟**（3 evals），预算太小，且 b2 里 agent 只改进 0.3% |
| `docs/results/results.md` | vit_tiny 5/5，mean **+42.44% ± 4.51%** | 数字很漂亮，但引用的 `runs/sweep/` **目录已不存在**，无法复核 |
| `runs/run_agent/run_iteration_*` | 22 个历史 run | objective 量级混乱（17.95 / 4953 / 8.46e5 / 1.1e13 混杂），是不同硬件点 + 不同时期代码，**不可横向比较** |
| §5.3 C1–C6 | ✅ 可复现 | `ablation_evidence.py` 确定性输出，**唯一能直接用的一批数字** |

### 补跑清单（按优先级）

1. **统一种子与预算的基线矩阵**：`baselines.runner --algo rs,sa,bo,ga,sweep
   --workload vit_tiny,vit_small,vit_base,mobilevit --budget 25 --seeds 0..9`
   → 这是 §5.2 的地基，最耗时，**先启动**
2. **智能体同预算矩阵**：4 workload × 10 seeds，`--iterations 25`
3. **§5.3 开关消融**：6 组 × 5 seeds × 15 evals
4. **§5.4 多模型**：3 模型 × 4 workload × 5 seeds
5. **§5.3 角色消融**：需先加 `--disable-stage`（我可以实现）

跑完用 `python -m baselines.compare` 一键出表和图。

---

## 7. 一页纸的节标题建议

```
5. Evaluation
  5.1 Experimental Setup
      5.1.1 Agent Runtime and Hardware
      5.1.2 Simulator Extensions
      5.1.3 LLM Backbones
      5.1.4 Baselines and Fairness Contract
      5.1.5 Benchmarks and Budget          [Table 1]
  5.2 Comparison with Traditional Optimizers  [Table 2, Fig. 2]
  5.3 Ablation Study
      Contributions C1–C6                  [Table 3]
      Runtime Switches                     [Table 4]
  5.4 Model-Agnosticism and Failure Modes     [Table 5]
```

一页装得下：Table 1（5 行）+ Table 2（压缩成 4 workload × 6 method）+ Fig. 2（1 张）
+ Table 3（6 行）+ Table 5（4 行）。Table 4 和失败模式表建议挪到附录。
