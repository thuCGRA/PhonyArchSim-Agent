# photonic-tuner

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
- [命令行参数](#命令行参数)
- [HTTP 服务模式](#http-服务模式)
- [对比基线](#对比基线)
- [实验与分析脚本](#实验与分析脚本)
- [运行测试](#运行测试)
- [仓库结构](#仓库结构)
- [第三方代码](#第三方代码)

---

## 特性

- **五阶段智能体流水线**：模型层 → 架构层 → 拓扑层 → 器件层 → 求解层，各角色职责
  单一，顺序由 Python 宿主控制，而非由模型决定。
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
model  ->  architecture  ->  topology  ->  device  ->  solution  ->  commit
```

| 阶段 | 职责 |
| --- | --- |
| **model（模型层）** | 从全模型证据中找出全局最关键的瓶颈层 |
| **architecture（架构层）** | 把每个候选层的成本归因到计算 / 存储 / 通信 / 缓冲 / 器件等根因 |
| **topology（拓扑层）** | 诊断光链路：最长光路径的插入损耗预算、光功率预算（激光输出 vs. 路径损耗 vs. 探测器灵敏度） |
| **device（器件层）** | 挑选具体的器件库变体 |
| **solution（求解层）** | 结合前几阶段结论与 RAG 检索到的历史案例，产出下一个合法硬件配置 |
| **commit（提交）** | 评估新配置、更新 Pareto 前沿与最佳点，决定继续下一轮或结束 |

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
  （审计 / 重放）；完成的运行会清理自己的检查点，保持库体量有界。

---

## 环境要求与安装

- **运行时解释器**：Python ≥ 3.11（`langgraph` 要求）。
- **真实 SimPhony 解释器**：Python 3.10（`torch 1.12.1`），独立于运行时。

### 一键复现环境（推荐）

克隆后，`.gitignore` 会把运行所需的 `.venv-*`、`.tools/`、`configs/llm.json` 排除在
版本库外；执行下面的脚本即可全部重建（幂等，可重复运行）：

```bash
bash scripts/env/bootstrap.sh                  # 同时创建 agent + simulator 两个环境
SKIP_SIMULATOR=1 bash scripts/env/bootstrap.sh # 只装 agent（快；不做 real 仿真时够用）
DRY_RUN=1 bash scripts/env/bootstrap.sh        # 先看会执行哪些步骤，不实际改动
```

脚本会创建：`.tools/`（uv 可执行 + uv 托管的 CPython 工具链）、`.venv-agent/`、
`.venv-simulator/`，以及从 `configs/llm.example.json` 复制的 `configs/llm.json`。
其中 `.tools/cache`、`.tools/mplconfig` 只是可随时清空的下载 / 字体缓存，不属于需要
复现的环境。需要精确版本时可加 `USE_AGENT_LOCK=1`（用
`scripts/env/requirements-agent.lock.txt`，含 CUDA torch 栈）。密钥只从
`OPENAI_API_KEY` 环境变量读取。

### 手动安装（等价步骤）

```bash
cd /path/to/epic        # 换成你的项目克隆目录

# 1) 创建运行时虚拟环境
.tools/uv venv .venv-agent --python 3.14

# 2) 安装运行时依赖
.tools/uv pip install --python .venv-agent/bin/python -r requirements.txt

# 3) （可选）语义嵌入需要 sentence-transformers；默认的 hashing 嵌入器不需要
.tools/uv pip install --python .venv-agent/bin/python "sentence-transformers>=2.7"
```

> 真实 SimPhony 环境（`.venv-simulator`）的搭建见 `scripts/env/setup_real.sh`
> （或用上面的 `bootstrap.sh` 一次装好），运行时通过 MCP 桥访问它，不在进程内导入。

---

## 快速开始

冒烟测试（确定性的假 LLM + 真实 SimPhony 仿真；无网络、无 API 密钥，但需要
`.venv-simulator`，且每次求值为分钟级）：

```bash
.venv-agent/bin/python main.py --provider mock --iterations 4
```

列出可用的工作负载：

```bash
.venv-agent/bin/python main.py --list-workloads
```

指定工作负载与初始硬件：

```bash
.venv-agent/bin/python main.py \
    --provider mock \
    --workload vit_tiny \
    --initial-hardware configs/initial_hardware.json \
    --iterations 6
```

用真实大模型跑（需先设置 `OPENAI_API_KEY`）：

```bash
.venv-agent/bin/python main.py \
    --provider paratera --model DeepSeek-V4-Pro-0813 \
    --nn-model-factory tests.factory:build_vit_tiny --workload vit_tiny
```

---

## API 密钥设置

密钥只从 `OPENAI_API_KEY` 环境变量读取（LangChain 约定）：项目**不会**从
`configs/llm.json`、命令行参数或任何凭据文件里读 key。离线冒烟测试
（`--provider mock`）完全不需要密钥。

创建该变量用辅助脚本 `configs/api_keys.py`。注意：子进程无法修改父 shell 的环境
变量，所以脚本只**打印** `export` 语句，必须用 `eval` 包住才会真正生效。

### 三种设置方式

```bash
# 1) 交互输入（推荐，key 不进入 shell 历史）
eval "$(python3 configs/api_keys.py)"

# 2) 命令行直接给（注意：key 会留在 shell 历史里）
eval "$(python3 configs/api_keys.py --key sk-你的key)"

# 3) 写死在脚本里，之后无需再输入（不推荐）
#    打开 configs/api_keys.py，把 API_KEY 槽位改成你的 key
eval "$(python3 configs/api_keys.py)"
```

方式 1 会提示 `Paste your OPENAI_API_KEY (input hidden):`，把 key 粘进去回车即可
（不回显）。

### 其他开关

```bash
python configs/api_keys.py --check             # 查状态（脱敏）：已设置 exit 0，未设置 exit 1
python configs/api_keys.py --masked            # 只输出脱敏值
eval "$(python configs/api_keys.py --unset)"   # 撤销当前终端里的变量
```

### 三个注意事项

1. **必须加 `eval`**：不加时脚本只把 `export OPENAI_API_KEY=...` 这行文字打在屏幕
   上，变量并不会被设置。根本原因是子进程改不了父 shell 的环境变量。
2. **只对当前终端有效**：新开终端或关掉窗口后变量消失，需要重新 `eval`。若想每次
   自动生效，可把 `export OPENAI_API_KEY=...` 写进 `~/.bashrc`，但这等于把明文密钥
   落到磁盘文件里，安全性不如每次手动设置。
3. **必须在运行 `main.py` 之前设置**，顺序反了会报：

```
   ValueError: OPENAI_API_KEY is not set.  Create it in this terminal first:
       eval "$(python configs/api_keys.py)"
```

> `configs/api_keys.py` 被 `.gitignore` 忽略，因为它的 `API_KEY` 槽位可能存放真实
> 密钥。脚本需要在 bash 终端里执行（`eval` 是 POSIX shell 语法）。

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
| `api_keys.py` | API 密钥设置辅助脚本（本地，被 gitignore） |

LLM 配置示例（`configs/llm.json`）：

```json
{
  "provider": "paratera",
  "model": "DeepSeek-V4-Pro-0813",
  "base_url": "https://llmapi.paratera.com/v1",
  "retry_attempts": 2
}
```

`provider` 内置预设：`mock`（离线假模型）、`paratera`、`paratera-flash`、
`deepseek`、`deepseek-official`。传入其他名字时不套用预设，此时必须同时给出
`model` 与 `base_url`（即可接入任意 OpenAI 兼容端点）。

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

器件候选由运行时从 `<simphony-root>/configs/devices` 自动发现；`search_space.json`
的 `candidates` 是当前器件库的快照。物理上不可实现的理想化 / 占位变体（如
`Customized_PS`、理想 `LT_Laser`）由 `EXCLUDED_SEARCH_VARIANTS` 从可搜索面隐藏。

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
- `--provider NAME`：LLM provider 覆盖（`mock` / `paratera` / `paratera-flash` /
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
  示例：`main.py --search-strategy sequential`
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
│   ├── metrics.json          # latency_s / energy_pj / area_um2 / objective
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
| `scripts/experiments/ablation_evidence.py` | 确定性离线消融证据（验证各贡献点） |
| `scripts/analysis/analyze_convergence.py` | 逐迭代收敛曲线与硬件差异 |
| `scripts/analysis/analyze_sweep.py` | 扫描结果统计、跨 run 确定性检查、LLM vs 回退归因 |
| `scripts/analysis/correct_memory_energy.py` | 修正 SimPhony 片上存储能量多计（`/ block_size`） |

---

## 运行测试

单一入口；当 `langgraph` 可导入时自动并入 `tests/deep/`：

```bash
.venv-agent/bin/python tests/run_tests.py    # 全量：领域 + deep（353 项）
python3 tests/run_tests.py                   # 仅领域测试（缺依赖时自动跳过 deep）
```

当前全量 **353 项**（领域 235 + deep 118）。离线端到端测试依赖
`FakeToolCallingModel`，结果确定可复现。

> 领域测试除标准库与 PyYAML 外，部分用例还会导入 `pydantic` / `tenacity`
> （经 `photon_tuner.agents.evaluate` 与 `photon_tuner.llm.model`）。若用裸
> `python3` 跑出现 `ModuleNotFoundError`，直接改用 `.venv-agent/bin/python`。

CI（`.github/workflows/ci.yml`）在 Python 3.11 / 3.12 上执行
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
