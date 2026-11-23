# OPEarch-Agent

一个面向**光子**深度学习加速器的 LLM 多智能体、RAG 辅助设计空间探索（DSE）框架。

它基于 SimPhony 仿真器，对架构参数、光学核心几何尺寸与器件库选型进行联合调优，
优化目标为归一化指标 `latency × energy-per-MAC × area`（延迟 × 单位 MAC 能耗 ×
面积）。运行时基于**原生 LangChain + LangGraph** 构建。

---

## 目录

- [特性](#特性)
- [架构](#架构)
- [环境要求与安装](#环境要求与安装)
- [快速开始](#快速开始)
- [API 密钥设置](#api-密钥设置)
- [配置说明](#配置说明)
- [搜索空间](#搜索空间)
- [指标单位](#指标单位)
- [动态调度](#动态调度)
- [命令行参数](#命令行参数)
- [HTTP 服务模式](#http-服务模式)
- [对比基线](#对比基线)
- [实验与分析脚本](#实验与分析脚本)
- [运行测试](#运行测试)
- [仓库结构](#仓库结构)
- [第三方代码](#第三方代码)

---

## 特性

- **五阶段智能体流水线**：瓶颈层 → 架构层 → 拓扑层 → 器件层 → 候选层，各角色职责
  单一，顺序由 Python 宿主控制，而非由模型决定。
- **动态工作流**：宿主根据先验成熟度、证据年龄及最近的目标改善，在 serial、pipeline 和 focus 之间逐轮选择。
- **原生 LangGraph 运行时**：用 `create_react_agent` 构建每个智能体，`StateGraph` 编排
  父图，`SqliteSaver` 做检查点——崩溃后可从中断处恢复。
- **证据评估门**：8 行验收协议对每个提案做一致性校验，被拒提案带着有界重试次数反馈
  给求解智能体。
- **RAG 历史复用**：SQLite 向量存储（实现 LangChain `VectorStore` / `BaseRetriever`
  接口），从历史调优案例中检索相似瓶颈与解法。
- **双通道检索**：瓶颈描述 + 硬件指纹联合打分，检索结果带相似度阈值过滤。
- **确定性可复现**：冒烟运行使用 `FakeToolCallingModel`，无网络、无 API 密钥，结果
  跨运行可复现。
- **双解释器分离**：真实 SimPhony 栈钉在 Python 3.10（通过 MCP 桥进程外运行），
  LangGraph 运行时需要 Python ≥ 3.11。
- **可观测性**：JSONL 链路追踪、调用树、逐阶段耗时、运行指标与状态变更日志。
- **传统算法基线**：`baselines/` 提供 RS / SA / BO / GA / 坐标扫描五种非 LLM 算法，
  与智能体共用同一评估器、搜索空间与目标函数，差异只来自决策策略。
- **HTTP 服务**：可选 FastAPI 封装，把调优暴露为 `POST /tune`。

---

## 架构

### 五阶段流水线

```
bottleneck  ->  architecture  ->  topology  ->  device  ->  candidate  ->  commit
```

| 阶段 | 职责 |
| --- | --- |
| **bottleneck（瓶颈层）** | 从全模型证据中找出全局最关键的瓶颈层 |
| **architecture（架构层）** | 把每个候选层的成本归因到计算 / 存储 / 通信 / 缓冲 / 器件等根因 |
| **topology（拓扑层）** | 诊断光链路：最长光路径的插入损耗预算、光功率预算（激光输出 vs. 路径损耗 vs. 探测器灵敏度） |
| **device（器件层）** | 挑选具体的器件库变体 |
| **candidate（候选层）** | 结合前几阶段结论与 RAG 检索到的历史案例，产出下一个合法硬件配置（一个一致的候选设计点） |
| **commit（提交）** | 评估新配置、更新 Pareto 前沿与最佳点，决定继续下一轮或结束 |

默认使用 `--schedule-strategy rag`，在每轮开始前选择工作流。指定 `--schedule-strategy fixed --schedule-mode serial` 可固定为串行对照。详见 [动态调度设计](docs/design/adaptive_scheduling.md)。
自适应模式下被跳过的阶段不调用模型，只把其最近一次诊断接续进上下文并标注 `evidence_age`。

每个阶段都是一个已编译的 LangGraph 智能体：模型在工具循环中收集证据，最终以
Pydantic 结构体（`structured_response`）产出结果。父 `StateGraph` 在每个节点后把
序列化的迭代上下文写入检查点，因此崩溃的运行可以用 `tune(resume_run_dir=...)`
从最后一个完成的阶段恢复。

### 关键组件

- **仿真层**：`SimulatorAdapter` 执行真实 SimPhony（`ONNArchSimulator`），经 MCP
  stdio 桥在 `.venv-simulator`（Python 3.10）中进程外运行。库内另保留一个解析镜像
  （`simulators/analytic.py`），仅供离线单元测试与实验脚本直接调用，不作为智能体的
  执行模式。
- **工具箱**：`toolbox/` 提供领域工具处理器（分析 / 器件），`toolbox/bindings/`
  把它们包装成 LangChain `BaseTool`，供各智能体的允许清单绑定。
- **RAG 存储**：`rag/vector_store.py` 实现历史案例的写入、双通道检索与 LangChain
  retriever 接口。
- **检查点**：两个 SQLite 库——父图运行检查点（可恢复）与智能体子图检查点
  （审计 / 重放）；完成的运行会清理自己的检查点记录。

---

## 环境要求与安装

支持 Linux x86_64（包括 Windows 上的 WSL2 Ubuntu）。首次安装需要联网获取 Python、CPU 版 PyTorch 及固定提交的仿真依赖，并使用 GNU make 和 C++ 编译器编译 CACTI。在 Ubuntu 上可先安装系统工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates g++ make libgl1 libglib2.0-0
```

将下面的仓库地址替换为自己的 GitHub 地址。在 Ubuntu 或 WSL 的 Bash 中运行：

```bash
git clone https://github.com/OWNER/REPOSITORY.git epic
cd epic
bash scripts/env/bootstrap.sh
```

上述命令克隆远端默认分支。若发布在其他分支，可在 `git clone` 后添加 `--branch 分支名`。后续命令都在克隆目录中执行。

统一配置入口是 `scripts/env/environment.conf`。它指定两个解释器、各自的依赖清单和上游源码版本。脚本创建 `.venv-agent/`（默认 Python 3.14）和 `.venv-simulator/`（Python 3.10、CPU PyTorch 1.12.1），编译 CACTI，并从示例生成本地 `configs/llm.json`。迁移时在目标机器执行同一命令。仿真器随后通过 MCP 子进程运行。

`SKIP_SIMULATOR=1` 适合只运行 Agent 侧单元测试；完整搜索需要两个环境。`USE_AGENT_LOCK=1` 使用已有的 Python 3.14 Agent 锁文件（含 CUDA 包）。默认安装使用 `requirements.txt` 和离线 hashing 嵌入器。`DRY_RUN=1` 可预览安装步骤。

---

## 快速开始

先列出工作负载，再用 ViT-Tiny 运行一次搜索。两条搜索命令分别使用独立的输出目录。

```bash
.venv-agent/bin/python main.py --list-workloads
.venv-agent/bin/python main.py --provider mock \
    --workload vit_tiny --nn-model-factory tests.factory:build_vit_tiny \
    --iterations 1 --output-root runs/smoke-mock
```

这一步使用本地确定性模型，初始设计和候选设计都交给真实 SimPhony 评估。运行下面的命令可用 Cloudmist 的 Flash 模型验证同一条流程。请在提示处输入自己的 Cloudmist 密钥。一次搜索迭代包含多个阶段和模型请求，按供应商用量计费。

```bash
read -rsp 'Cloudmist API key: ' OPENAI_API_KEY && export OPENAI_API_KEY
printf '\n'
.venv-agent/bin/python main.py --provider cloudmist \
    --base-url https://v2.cloudmist.cloud/v1 --model deepseek-v4.1-flash \
    --workload vit_tiny --nn-model-factory tests.factory:build_vit_tiny \
    --iterations 1 --output-root runs/smoke-cloudmist
```

首次运行时，输出位于各自的 `run_iteration_001/` 目录。检查以下文件即可核验流程。

| 相对运行目录的文件 | 验收内容 |
| --- | --- |
| `iter_000/metrics.json` | 初始设计的仿真指标，`objective` 为有限数值 |
| `iter_001/metrics.json` | 本轮候选设计的仿真指标，可和初始值对比 |
| `iter_001/analysis/schedule.json` | 本轮调度决定。空历史下默认从 `serial` 开始 |
| `iter_001/analysis/llm_trace.jsonl` | 本轮的模型交互记录。结合启动命令中的 provider 和端点核对实际调用 |

一轮验收检查模型调用、候选生成和仿真评估能否走通。综合目标越小越好，单轮候选可能优于或劣于初始设计。

---

## API 密钥设置

真实 LLM 调用从 `OPENAI_API_KEY` 环境变量读取密钥。使用 Bash 的隐藏输入可在本终端设置：

```bash
read -rsp 'API key: ' OPENAI_API_KEY && export OPENAI_API_KEY
printf '\n'
```

也可运行 `eval "$(python3 configs/api_keys.py)"` 进行交互设置，或用 `python3 configs/api_keys.py --check` 检查当前终端。`--provider mock` 可以在没有密钥的情况下验证完整仿真流程。

---

## 配置说明

`configs/` 目录存放运行所需的非敏感配置：

| 文件 | 用途 |
| --- | --- |
| `llm.example.json` | LLM 配置模板（provider / model / base_url / retry 等），复制为 `llm.json` 使用 |
| `llm.json` | 本地 LLM 配置（被 gitignore，可能含环境相关设置） |
| `initial_hardware.json` | 初始硬件配置模板 |
| `initial_hardware_mesh.json` | 网格（mesh）拓扑的初始硬件 |
| `initial_hardware_v2.json` | v2 初始硬件 |
| `search_space.json` | 分层搜索空间候选集（architecture / topology / device / device_param） |
| `workloads.json` | 工作负载定义（`vit_tiny` / `vit_small` / `vit_base`） |
| `prompts/prompts.yaml` | 各智能体的提示词模板（YAML） |
| `api_keys.py` | API 密钥设置辅助脚本（隐藏输入或读取当前环境变量） |

LLM 配置示例（`configs/llm.json`）：

```json
{
  "provider": "paratera",
  "model": "DeepSeek-V4-Pro-0813",
  "base_url": "https://llmapi.paratera.com/v1",
  "retry_attempts": 2
}
```

`provider` 内置预设：`mock`（离线假模型）、`cloudmist`、`paratera`、
`paratera-flash`、`deepseek`、`deepseek-official`。传入其他名字时不套用预设，
此时必须同时给出 `model` 与 `base_url`（即可接入任意 OpenAI 兼容端点）。

如已将 Cloudmist 凭据保存在本地 `.env` 中，可在仓库根目录运行 `source scripts/env/activate_cloudmist.sh`，再执行“快速开始”中的 Cloudmist 命令。`.env` 供本机使用，应设置为仅本人可读写。公开仓库保留配置示例和密钥读取代码。

端点解析优先级：`--base-url` / `llm.json` 的 `base_url` > `OPENAI_BASE_URL`
环境变量 > provider 预设。

---

## 搜索空间

搜索空间候选值**只声明一次**：[`configs/search_space.json`](configs/search_space.json)
（分层格式，`version: 2`）。唯一的 Python 实现是
[`photon_tuner/core/search_space.py`](photon_tuner/core/search_space.py)，它在导入时
读取该 JSON 得到 `LAYERED_DESIGN_SPACE`，扁平回退空间 `DEFAULT_ARCH_SEARCH_SPACE`
也由同一份规格派生——不再有第二份候选列表。这条契约由
`tests/test_search_space_single_source.py` 用 AST 扫描强制守住。

两处候选**不在** JSON 里声明（避免漂移）：

- `core_kind` / `node_kind` —— 由拓扑注册表
  [`topology_space.py`](photon_tuner/core/topology_space.py) 的 `CORE_KINDS` /
  `NODE_KINDS` 注入（那里同时承载 `real` / `evaluable` 标志、core→node 耦合矩阵与
  器件槽位）。JSON 若重复声明会直接报错。
- 器件变体 —— 运行时从 SimPhony 器件库 YAML 发现；JSON 里的清单只是文档快照。

分层与三个联合搜索 agent 一一对应：

| 层 | 对应 agent | 搜索参数 |
| --- | --- | --- |
| `architecture` | architecture | `tiles`、`cores_per_tile`、`dataflow`、`precision_in_bit` / `precision_w_bit` / `precision_out_bit`、`hbm_bandwidth`、`glb1_cache_size` / `glb2_cache_size` / `rf_cache_size`、`core_height`、`core_width`、`num_wavelength`、`work_freq` |
| `topology` | topology | `core_kind`、`node_kind`、`sharing_policy`、`temporal_accum`、`laser_distribution`、`mesh_rank` |
| `device` | device | 器件库变体选择（按 `device_type` 逐类）＋ `laser_source` / `splitter_kind` 实例级开关 |
| `device_param` | —（固定层） | 库内物理参数；只做变体选择，不搜索 |

参数规格字段：

- `map`：从 `hardware` 字典根到该值的路径（如 `["arch", "optical_core", "work_freq"]`）。
  **层名是“搜索分组”，与硬件字典键（`arch.system`、`arch.optical_core`）相互独立**：
  光学核心几何虽归 `architecture` 层，其值仍存放在 `arch.optical_core` 下。
- `values` / `step`：粗粒度候选网格与离散步进。
- `refine`：是否允许自适应细化（围绕当前设计点收窄网格、细分步进）。
- `constraint`：跨参数可行性约束，例如 `work_freq <= mzm.symbol_rate`、
  `in_bit <= dac.prec`、`out_bit <= adc.prec`。

### 网格规则（两条硬约束）

1. **架构层的几何 / 内存 / 频率轴一律取 2 的幂，上限至少到 64。**
   例如 `tiles ∈ {1,2,4,8,16,32,64}`、`core_height ∈ {8,16,32,64}`、
   `work_freq ∈ {2,4,8,16,32,64}`、缓存 `∈ {512,1024,2048,4096}` KB。
   目的是让同一拓扑族下有足够密的邻域可走，同时保持每个档位可解释（LLM 才能在
   载荷里做有意义的比较）。精度轴保持 `{2,4,8}` bit：上界由 DAC/ADC 的 `prec`
   与采样率链物理给定，扩到 64 bit 不可仿真。
2. **三级缓存下界是 512 KB，而不是 64 KB。** CACTI 在当前的
   `block_size` / associativity 下对 64/128/256 KB 直接报
   `Invalid memory configuration provided for GLB1`。这是**后端建模边界**，不是
   物理上不存在——把数字加回候选表只会让每次求值抛错。要搜索更小缓存，需要更换
   存储模型或放宽 associativity 约束。

> 提示：`work_freq` 的高档位在当前基线器件链下会被 `normalize` 钳到
> `min(MZM 符号率, DAC 采样率, ADC 采样率)`。这是有意的——器件升级后同一档位即可
> 生效，搜索空间不需要随器件而改写。

### 参数可见性

`searchable_parameters` 工具以**完整设计空间**为准枚举每个参数（22 个），并按当前
拓扑族**与求值后端**标注 `effective` / `superseded_by` / `not_applicable` /
`fixed_by_design` / `not_consumed`。它同时返回 `effect_notes`——每个轴主要移动哪个
目标因子、以及主要陷阱。

两类"看不见的轴"都曾导致搜索退化，现在都有回归测试：

* **拓扑族级**：早期只枚举了 `PARAM_SCOPE` 里按族分列的 5 个参数，智能体以为架构层
  只剩 4 根旋钮、拓扑层零可搜索轴。
* **后端级（死轴）**：`hbm_bandwidth` 与 `precision_w_bit` 在 real 模式下**完全不参与
  计算**（各取全部候选值得到逐位相同的 latency/energy/area），却曾被报告为可搜索。
  它们只在解析镜像里生效——两条求值路径消费的参数集不同，仅靠域测试发现不了。
  现在由 `MODE_INERT_PARAMS` 声明，工具与基线动作空间都据此排除。

器件候选由运行时从 `<simphony-root>/configs/devices` 自动发现；`search_space.json`
的 `candidates` 是当前器件库的快照。物理上不可实现的理想化 / 占位变体（如
`Customized_PS`、理想 `LT_Laser`）由 `EXCLUDED_SEARCH_VARIANTS` 从可搜索面隐藏。

---

## 指标单位

全部指标使用光电系统惯例量纲，换算集中在
[`photon_tuner/core/units.py`](photon_tuner/core/units.py) 一处：

| 键 | 单位 | mobilevit 基线 |
|---|---|---|
| `latency_ms` | ms | 4.77 |
| `energy_mj`（计算+存储） | mJ | 9.53 |
| `energy_compute_mj` / `energy_memory_mj` | mJ | 1.71 / 7.83 |
| `area_mm2`（逐器件+节点边界框） | mm² | 77.75 |
| `fJ_per_mac` | fJ/MAC | 926 |

三项换算因子之积为 1，因此**目标函数数值在单位换算前后完全相同**。报告里同时保留
`si_metrics`（SI 镜像）与 `units`（单位声明）供核对。

两条更重要的**口径**修正（详见 [docs/design/units.md](docs/design/units.md)）：

* **能耗 = 计算 + 存储**。`simu_energy` 只给计算能耗，而延时却含存储延时；实测存储
  占总能耗 75–90%。只计计算能耗会让优化方向系统性偏移（实测会选出不同的最优设计点）。
* **面积 = 逐器件 + 节点边界框**。厂商 `simu_area` 的过滤条件误写成
  `if "node" not in v`（判断字典内部键而非键名），导致节点面积被重复计入、真实模式
  面积多算约 43%。聚合层已按正确口径重算，原始值保留在 `area.vendor_total_um2`。

---

## 动态调度

默认的 `--schedule-strategy rag` 由宿主程序读取有效案例数、案例覆盖的设计层、各层证据年龄和历史最优目标值，逐轮选择工作流。

| 模式 | 默认触发条件与工作方式 |
|---|---|
| `serial` | 先验尚未积累 3 个有效案例或尚未覆盖架构、拓扑、器件三层时，依次执行各层，生成本轮证据。需要启动或恢复流水线但证据过期时，同样通过串行搜索刷新证据。 |
| `pipeline` | 先验成熟且三层证据年龄均不超过 2 轮时建立流水线。各层按 0、1、2 轮的偏移使用证据。 |
| `focus` | 至少完成过一轮流水线，且最近 10 轮的历史最优值每轮改善均不超过 10% 时，集中搜索一个设计层。出现超过阈值的改善后重新判断流水线条件。 |

固定工作流使用 `--schedule-strategy fixed --schedule-mode serial|pipeline|focus`。详细判据见 [调度实现](photon_tuner/runtime/scheduler.py)。

---

## 命令行参数

```
usage: photon-tuner [--initial-hardware FILE] [--search-space FILE]
                    [--simphony-root DIR] [--devicelib-root DIR]
                    [--output-root DIR] [--iterations N]
                    [--llm-config FILE]
                    [--provider NAME] [--model NAME] [--base-url URL]
                    [--model-type transformer|cnn] [--nn-model-factory F]
                    [--workload NAME] [--list-workloads] [--il-budget DB]
                    [--state-strategy best|next]
                    [--search-strategy joint|sequential] [--topology-sweep]
                    [--no-gate] [--gate-retries N] [--no-trajectory]
                    [--quiet] [--verbose] [--serve] [--serve-host H]
                    [--serve-port P]
```

以下为全部参数，示例统一以 `.venv-agent/bin/python main.py` 为前缀，简写为 `main.py`。

**输入 / 输出**

- `--initial-hardware FILE`：初始光子硬件配置的 JSON 文件，作为调优起点。
  示例：`main.py --initial-hardware configs/initial_hardware.json`
- `--search-space FILE`：覆盖架构 / 器件搜索空间候选集的 JSON 文件。
  示例：`main.py --search-space configs/search_space.json`
- `--output-root DIR`：每轮运行产物、历史库与检查点的输出目录，默认 `runs/run_agent/`。
  示例：`main.py --output-root runs/run_agent`

**仿真**

- 仿真模式：**固定为 real**。没有 `--mode` 选项——智能体始终经 MCP 桥调用
  `.venv-simulator` 中的真实 `ONNArchSimulator`。解析镜像仅保留在库内部供单元测试
  与实验脚本直接调用，不是智能体的执行模式。
- `--simphony-root DIR`：SimPhony 项目根目录（默认指向 vendored 副本）。
  示例：`main.py --simphony-root photon_tuner/simulators/simphony`
- `--devicelib-root DIR`：器件库根目录，默认 `<simphony-root>/configs/devices`。
  示例：`main.py --devicelib-root /data/devices`
- `--model-type {transformer,cnn}`：real 模式下用于映射配置的模型族
  （BERT/MobileViT → transformer），默认 `transformer`。
  示例：`main.py --model-type cnn`
- `--nn-model-factory F`：real 模式下构建 ONN/nn 模型的 `'module:function'`。
  示例：`main.py --nn-model-factory tests.factory:build_vit_tiny`
- `--workload NAME`：工作负载名（`vit_tiny` / `vit_small` / `vit_base`），默认取
  `workloads.json` 的 `default` 项（`vit_tiny`）。
  示例：`main.py --workload vit_tiny`
- `--list-workloads`：列出所有已注册的工作负载后退出。
  示例：`main.py --list-workloads`
- `--il-budget DB`：光链路插入损耗预算（dB），超出该预算的最长光路径会被软惩罚。
  示例：`main.py --il-budget 8.0`

> `--workload` 与 `--nn-model-factory` 会分别写入 `SIMPHONY_WORKLOAD` /
> `SIMPHONY_NN_FACTORY`，由 MCP 服务端读取；两者需与所选模型族保持一致。

**LLM**

- `--llm-config FILE`：LLM 配置 JSON（provider/model/base_url 等），默认
  `configs/llm.json`。API 密钥不在此文件，始终取自 `OPENAI_API_KEY`。
  示例：`main.py --llm-config configs/llm.json`
- `--provider NAME`：LLM provider 覆盖（`mock` / `cloudmist` / `paratera` / `paratera-flash` /
  `deepseek` / `deepseek-official`，或自带 `model` + `base_url` 的 OpenAI 兼容端点）。
  示例：`main.py --provider mock`
- `--model NAME`：LLM 模型名覆盖。
  示例：`main.py --model DeepSeek-V4-Pro-0813`
- `--base-url URL`：LLM 端点 base URL 覆盖。
  示例：`main.py --base-url https://your-endpoint/v1`

**搜索策略**

- `--iterations N`：调优迭代轮数，默认 4。
  示例：`main.py --iterations 6`
- `--state-strategy {best,next}`：下一轮继承哪个已评估点——`best`（迄今最优）或
  `next`（最新提案）。默认 `best`。
  示例：`main.py --state-strategy next`
- `--search-strategy {joint,sequential}`：联合搜索方式——`joint`（每轮所有层一起）
  或 `sequential`（架构 → 拓扑 → 器件轮转）。默认 `joint`。
  固定轮转示例：`main.py --schedule-strategy fixed --search-strategy sequential`
- `--schedule-strategy {rag,fixed}`：默认通过 `rag` 动态调度，`fixed` 用于固定工作流对照。
- `--schedule-mode {serial,pipeline,focus}`：指定固定工作流的运行模式。
  示例：`main.py --schedule-strategy fixed --schedule-mode pipeline`
- `--topology-sweep`：预先评估每个可评估的电路模板（`core_kind`）并归档到 Pareto
  前沿。
  示例：`main.py --topology-sweep`
- `--no-gate`：禁用证据评估门。
  示例：`main.py --no-gate`
- `--gate-retries N`：证据门拒绝解决方案后的最大重提案次数，默认 2。
  示例：`main.py --gate-retries 3`
- `--no-trajectory`：不写入 `trajectory.json`。
  示例：`main.py --no-trajectory`

**输出 / 日志 / 服务**

- `--quiet`：抑制进度输出。
  示例：`main.py --quiet`
- `--verbose`：开启 DEBUG 级日志（等价于 `PHOTON_TUNER_LOG_LEVEL=DEBUG`）。
  示例：`main.py --verbose`
- `--serve`：改为运行 FastAPI HTTP 服务（`POST /tune`、`GET /health`），而非一次性
  调优。
  示例：`main.py --serve`
- `--serve-host H`：`--serve` 时的绑定主机，默认 `0.0.0.0`。
  示例：`main.py --serve --serve-host 127.0.0.1`
- `--serve-port P`：`--serve` 时的端口，默认 `8000`。
  示例：`main.py --serve --serve-port 8080`

组合示例（完整一次调优，假 LLM、真实仿真）：

```bash
.venv-agent/bin/python main.py \
    --provider mock \
    --workload vit_tiny \
    --initial-hardware configs/initial_hardware.json \
    --search-space configs/search_space.json \
    --iterations 6 --il-budget 8.0 \
    --state-strategy best --search-strategy joint \
    --output-root runs/run_agent
```

### 运行产物

每次运行在 `--output-root` 下生成一个 `run_iteration_NNN/` 目录：

```
run_iteration_NNN/
├── iter_000/                 # 基线点（未改动的初始硬件）
│   ├── hardware.json         # 该轮硬件配置
│   ├── workload.json         # 工作负载
│   ├── metrics.json          # latency_ms / energy_mj / area_mm2 / objective
│   ├── report.json           # SimPhony 原始报告
│   ├── area_detail.json  energy_detail.json  insertion_loss.json  memory.json
│   └── analysis/             # 各阶段智能体产出
├── iter_001/ ...             # 后续迭代
├── pareto_frontier.json      # Pareto 前沿归档
├── summary.json              # 汇总（best / 迭代数 / run_dir）
└── trajectory.json           # 轨迹（可 --no-trajectory 关闭）
```

---

## HTTP 服务模式

安装可选的 `fastapi` / `uvicorn` 后，可以把调优暴露为 HTTP 服务：

```bash
pip install fastapi uvicorn
.venv-agent/bin/python main.py --serve --serve-host 0.0.0.0 --serve-port 8000
```

接口：

- `GET /health` → `{"status": "ok"}`
- `POST /tune` → 请求体示例：

```json
{
  "initial_hardware": {},
  "iterations": 4,
  "state_strategy": "best",
  "search_strategy": "joint",
  "il_budget": null,
  "topology_sweep": false
}
```

响应包含 `iterations`、`best`（归一化目标）、`best_hardware`、`run_dir` 等字段。

---

## 对比基线

`baselines/` 提供可复现的传统算法基线，用于与智能体做公平对比。它**只读**复用
`photon_tuner` 的评估器、搜索空间与目标函数，不修改后者代码。

| 算法 | 说明 |
| --- | --- |
| `rs` | 随机搜索（最优点邻域随机游走） |
| `sa` | 模拟退火（log-objective 空间的几何冷却） |
| `bo` | 贝叶斯优化（Matern(ν=5/2) GP + EI） |
| `ga` | 遗传算法（分组交叉 + 逐基因变异 + 精英保留） |
| `sweep` | 坐标扫描 / 单参数穷举（确定性补充基线） |

公平性契约：所有方法走**同一评估器**（`SimphonyAdapter`）、**同一搜索空间**、
**统一动作空间**（最小合法编辑），预算统一按 SimPhony 求值次数计
（`--iterations N` 与 `--budget N` 都是 1 次初始求值 + N 次提案）。

```bash
# 单个基线：随机搜索，50 次提案（共 51 次求值）
.venv-agent/bin/python -m baselines.runner \
    --algo rs --workload vit_tiny --budget 50 --seeds 0 \
    --nn-model-factory tests.factory:build_vit_tiny

# 四主基线 × 多 workload × 多 seed
.venv-agent/bin/python -m baselines.runner \
    --algo rs,sa,bo,ga --workload vit_tiny,vit_small,vit_base \
    --budget 50 --seeds 0,1,2 --output-root runs/run_baseline \
    --nn-model-factory tests.factory:build_vit_tiny

# 生成 agent vs 基线对比报告（表格 + 收敛曲线 SVG）
.venv-agent/bin/python -m baselines.compare \
    --agent-runs "runs/run_agent/run_iteration_*" \
    --baseline-root runs/run_baseline \
    --out reports/compare
```

产物：`runs/run_baseline/<algo>/<workload>/seed_<N>/{trajectory.json,summary.json,iter_*/}`。
详见 [`baselines/README.md`](baselines/README.md)。

---

## 实验与分析脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/env/bootstrap.sh` | 一键重建 `.tools/` + 两个 venv + `configs/llm.json`（幂等） |
| `scripts/env/setup_real.sh` | 单独搭建真实 SimPhony 环境（Py3.10 + torch 1.12 + CACTI 编译） |
| `scripts/env/activate_{agent,real}.sh` | `source` 后激活对应环境 |
| `scripts/dev/probe_real.py` | 探针：进程内跑一次真实 SimPhony 求值 |
| `scripts/dev/smoke_mcp.py` | 走完整 MCP 跨进程链路求值一次 |
| `scripts/dev/smoke_tune.py` | 端到端 DSE 冒烟（1 次迭代，临时目录） |
| `scripts/experiments/sweep.py` | 批量顺序扫描（workloads × N runs），结果落 `runs/sweep/` |
| `scripts/experiments/verify_search_space.py` | 搜索空间有效性验证（单变量消融 → `data/evidence.db`） |
| `scripts/analysis/scheduler_sensitivity.py` | 调度阈值敏感性：用落盘观测量离线回放模式决策（不消耗评估），并自检是否逐轮复现真实轨迹 |
| `scripts/experiments/phel_experiments.py` | 两组对比实验（等预算 / 同目标收敛），每组产出一个文件夹内的「数据 + 大模型运行记录」两个文件 |
| `scripts/experiments/ablation_evidence.py` | 确定性离线消融证据（验证各贡献点） |
| `scripts/analysis/analyze_convergence.py` | 逐迭代收敛曲线与硬件差异 |
| `scripts/analysis/analyze_sweep.py` | 扫描结果统计、跨 run 确定性检查、LLM vs 回退归因 |
| `scripts/analysis/correct_memory_energy.py` | 修正 SimPhony 片上存储能量多计（`/ block_size`） |

---

## 运行测试

单一入口；当 `langgraph` 可导入时自动并入 `tests/deep/`：

```bash
.venv-agent/bin/python tests/run_tests.py    # 领域测试和 deep 测试
python3 tests/run_tests.py                   # 仅领域测试（缺依赖时自动跳过 deep）
```

离线端到端测试依赖
`FakeToolCallingModel`，结果确定可复现。

> 领域测试除标准库与 PyYAML 外，部分用例还会导入 `pydantic` / `tenacity`
> （经 `photon_tuner.agents.evaluate` 与 `photon_tuner.llm.model`）。若用裸
> `python3` 跑出现 `ModuleNotFoundError`，直接改用 `.venv-agent/bin/python`。

CI 的迁移验收任务通过安装脚本重建双环境，并使用 mock LLM 运行一轮真实 SimPhony 仿真。单元测试任务在 Python 3.11 / 3.12 上执行
`coverage run --source=photon_tuner tests/run_tests.py`，配置见 `.coveragerc`。

---

## 仓库结构

```
photon_tuner/
  __init__.py           # 包入口
  cli.py                # 命令行参数解析与 main() 入口
  server.py             # 可选 FastAPI HTTP 封装（POST /tune、GET /health）
  agents/               # LangGraph 智能体
    builders.py         #   按角色构建 react 智能体（工具白名单 → SystemMessage → agent）
    evaluate.py         #   证据评估门：8 行协议确定性校验提案
    outputs.py          #   各阶段 Pydantic 输出模型
    prompts.py          #   提示词模板加载
  bridge/               # 进程外 SimPhony MCP 桥
    mcp/
      simphony_server.py  # MCP stdio 服务器（在 Python-3.10 环境内跑 SimPhony）
      simphony_tool.py    # MCP 工具定义与 ping
  common/               # 横切基础设施
    checkpoint.py       #   LangGraph 检查点构建
    io.py               #   JSON 读写
    logging.py          #   日志级别配置
    metrics.py          #   运行指标聚合
    paths.py            #   项目根 / SimPhony 根等路径常量
    progress.py         #   进度报告
    tool_types.py       #   工具类型别名
    tracing.py          #   LLM 链路追踪收集器（调用树）
    truncate.py         #   负载截断与上下文预算
    utils.py            #   候选钳位、层分组、算子特征
  core/                 # 领域模型
    device_frontier.py  #   器件 Pareto 前沿
    objective.py        #   归一化目标（latency×energy×area）与 IL 预算惩罚
    pareto.py           #   Pareto 支配关系与归档
    provenance.py       #   器件来源审计与 DOI 校验
    schemas.py          #   评估 / 分析结果数据模型
    search_space.py     #   搜索空间唯一实现：规格、可行性校验、自适应细化、硬件操作
    units.py            #   规范单位换算（ms / mJ / mm² / fJ·MAC⁻¹），唯一换算点
    topology.py         #   拓扑规范应用（sharing / temporal / laser）
    topology_space.py   #   拓扑模板、耦合校验、候选过滤（core_kind/node_kind 唯一来源）
    topology_spec.py    #   拓扑节点 / 图规范
    workload.py         #   工作负载构建（ViT）
  evidence/             # 搜索空间有效性验证
    probe.py            #   单变量消融探针
    store.py            #   SQLite 证据存储
    sub_arch_map.py     #   搜索参数到子架构位置的映射
    workload_features.py  # 工作负载硬件指纹特征
  llm/                  # LLM 层
    cache.py            #   进程内响应缓存
    config.py           #   LLMConfig 数据类与校验
    model.py            #   Chat[OI] 构建、provider 预设、重试策略
  rag/                  # 检索增强
    embeddings.py       #   嵌入模型（hashing / sentence-transformers）
    vector_store.py     #   SQLite 历史向量存储（langchain VectorStore/Retriever 接口）
  runtime/              # 编排运行时
    orchestrator.py     #   DeepAgentTuner 编排器（五阶段循环、门控、检查点清理）
    payloads.py         #   类型化阶段负载
    scheduler.py        #   RAG 驱动的模式调度（纯函数：观测信号 → 模式/聚焦层/证据偏移）
    state.py            #   迭代上下文、快照 / 恢复、schema 版本
  simulators/           # 仿真层
    analytic.py         #   解析求值镜像（仅离线测试 / 实验直接调用）
    base.py             #   SimulatorAdapter 抽象接口
    mesh_model.py       #   MZI 网格解析模型
    netlist.py          #   网表装配（hardware → sub_arch）
    simphony.py         #   SimPhony 适配器（MCP 桥 / 真实报告）
    simphony/           #   vendored 第三方 SimPhony 副本（见“第三方代码”）
  testing/
    fake_model.py       #   FakeToolCallingModel（确定性假 LLM）
  toolbox/              # 领域工具与 LangChain 绑定
    tool_cache.py       #   工具结果缓存（FIFO 淘汰）
    analysis_tools/     #   分析工具处理器（edits / hardware_metrics / layer_analysis / search_state）
    bindings/           #   LangChain BaseTool 绑定层
    common_tool/        #   共享工具处理器（报告格式化、ToolContext）
    device_tools/       #   器件工具处理器（catalog / edits / frontier / metrics）
baselines/              # 传统算法基线（详见 baselines/README.md）
  action_space.py       #   统一动作空间
  evaluator.py          #   SimPhony 评估封装（指纹去重 + 求值计数）
  encoding.py           #   硬件特征编码（BO 的 GP 代理模型）
  harness.py            #   与算法无关的搜索循环
  algorithms/           #   rs / sa / bo / ga / sweep
  runner.py             #   命令行：跑基线
  compare.py            #   命令行：agent vs 基线对比报告
  plotting.py           #   零依赖 SVG 绘图
configs/                # 非敏感运行配置（见“配置说明”）
scripts/
  env/                  #   环境搭建 / 激活
  dev/                  #   开发冒烟
  experiments/          #   扫描、搜索空间验证、消融
  analysis/             #   结果分析
  misc/                 #   一次性小工具
tests/                  # 统一测试目录（详见 tests/README.md）
  run_tests.py          #   测试入口
  factory.py            #   real 模式模型工厂（tests.factory:build_<workload>）
  cnn_models.py         #   QVGG 纯 CNN 模型
  smoke_real.py         #   real 模式手动冒烟
  test_*.py             #   领域测试（235 项）
  deep/test_*.py        #   智能体运行时测试（118 项，需 langgraph）
docs/
  design/               #   设计与实现方案
  results/              #   实验结果与分析
  archive/              #   历史方案存档
main.py                 # 命令行入口（转发到 photon_tuner.cli）
```

---

## 第三方代码

[`photon_tuner/simulators/simphony/`](photon_tuner/simulators/simphony) 是
[ScopeX-ASU/SimPhony](https://github.com/ScopeX-ASU/SimPhony) 的 vendored 副本
（MIT 许可证；论文 arXiv:2411.13715）。详见其附带的
[`LICENSE`](photon_tuner/simulators/simphony/LICENSE) 与
[`VENDORED.md`](photon_tuner/simulators/simphony/VENDORED.md)。其中的 CACTI 子组件
来自 Hewlett-Packard（见 `cacti/` 下的头文件）。

真实 SimPhony 栈固定在 Python 3.10 / torch 1.12，因此通过 MCP 桥进程外运行，
不与主运行时共享解释器。
