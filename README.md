# OPEarch-Agent

**A Multi-Agent Framework for Cross-Layer Design Space Exploration of Optoelectronic AI Computing Architectures**

OPEarch-Agent 用于搜索光电 AI 计算架构的硬件设计。给定一个神经网络工作负载、一份初始硬件配置和设计空间，系统先用 SimPhony 评估初始设计，再依据仿真报告定位瓶颈。架构、拓扑和器件智能体分析各层可调整的参数，候选智能体整合提案并提交仿真。系统按轮保存候选硬件、智能体分析和仿真结果，供后续搜索和复核使用。

基础综合目标为 **时延（ms）× 每 MAC 能耗（fJ/MAC）× 芯片面积（mm²）**，数值越低越好。配置插入损耗预算后，超过预算的设计还会受到软惩罚。若要报告相对初始设计的改善率，使用 `(J_initial - J_candidate) / J_initial × 100%`。目标的计算和权重见 [objective.py](photon_tuner/core/objective.py)。

框架用 LangGraph 编排智能体，通过检索增强生成（RAG）复用已评估的历史案例。Python 调度器根据案例成熟度、历史证据年龄和最近的搜索收益，逐轮选择 `serial`、`pipeline` 或 `focus`。可更换提供兼容接口的语言模型，并在同一仿真器上运行传统搜索方法进行比较。

[安装](#环境要求与安装) · [运行第一轮](#快速开始) · [工作负载](#工作负载与实验) · [运行产物](#运行产物) · [调度](#动态调度) · [测试](#测试与迁移)

## 环境要求与安装

完整搜索支持 **Linux x86_64**，已在 Ubuntu 24.04 x86_64 环境验证。WSL2 是 Windows 用户运行 Ubuntu 的一种方式。按手头的机器选择执行位置：

| 你使用的机器 | 在哪里运行下面的命令 |
| --- | --- |
| x86_64 Ubuntu 工作站或服务器 | 直接在 Ubuntu 终端运行。 |
| Windows 或 macOS，有 Linux 服务器账号 | 在本机终端执行 `ssh 用户名@服务器地址`，登录 x86_64 Ubuntu 服务器后，在服务器上克隆仓库、安装环境和运行搜索。 |
| Intel/AMD x86_64 电脑，需要在本机运行 | 在 Ubuntu 24.04 x86_64 虚拟机的终端内操作。Windows 用户已有 WSL2 Ubuntu 时，也可直接使用它。 |
| Apple Silicon 或其他 ARM 设备 | 可通过 SSH 使用 x86_64 Ubuntu 服务器。当前仿真依赖清单指定了 Linux x86_64 的 CPU 版 PyTorch 安装包。 |

在选好的 Linux 终端执行 `uname -sm`，预期输出为 `Linux x86_64`。若只需核验安装和真实仿真是否跑通，仓库启用 GitHub Actions 后，也可查看 [CI 工作流](.github/workflows/ci.yml) 的 `real-simulator` 任务。该任务使用 mock 模型和真实 SimPhony，无需模型密钥。

首次安装需要联网下载 Python、CPU 版 PyTorch 和仿真依赖，并用 GNU make 和 C++ 编译器编译 CACTI。先在 Ubuntu 安装系统工具，在终端运行以下命令：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates g++ make libgl1 libglib2.0-0
```

完成后进入用cd命令进入/home/your username（否则会提示Permission denied）下接着运行：

```bash
git clone https://github.com/thuCGRA/PhonyArchSim-Agent.git 
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

这一步使用本地确定性模型，初始设计和候选设计都交给真实 SimPhony 评估。运行下面的命令可用 Cloudmist 的 Flash 模型验证同一条流程。随仓库提供的 [Cloudmist 配置](configs/llm.cloudmist.example.json) 固定了本例的模型、端点、请求长度上限和超时。请在提示处输入自己的 Cloudmist 密钥。一次搜索迭代包含多个阶段和模型请求，按供应商用量计费。

```bash
read -rsp 'Cloudmist API key: ' OPENAI_API_KEY && export OPENAI_API_KEY
printf '\n'
.venv-agent/bin/python main.py --provider cloudmist \
    --llm-config configs/llm.cloudmist.example.json \
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

可用下面的命令检查首次 Cloudmist 运行的结果。若验证的是 mock 示例，把路径中的 `smoke-cloudmist` 改为 `smoke-mock`。

```bash
.venv-agent/bin/python - <<'PY'
import json
import math
from pathlib import Path

run = Path("runs/smoke-cloudmist/run_iteration_001")
for step in ("iter_000", "iter_001"):
    value = json.loads((run / step / "metrics.json").read_text())["objective"]
    assert math.isfinite(value), (step, value)
    print(step, value)
print("mode", json.loads((run / "iter_001/analysis/schedule.json").read_text())["mode"])
PY
```

一轮验收检查模型调用、候选生成和仿真评估能否走通。单轮候选可能优于或劣于初始设计。

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

## 工作负载与实验

`--list-workloads` 列出默认配置中的 ViT-Tiny、ViT-Small 和 ViT-Base。论文实验另有一份[九负载清单](configs/paper_experiment_workloads.json)，涵盖 ViT（Tiny/Small/Base）、MobileViT（XXS/XS/S）和 CNN（Tiny/Small/Base）。每项给出模型类型、模型工厂及对应的初始硬件文件。运行指定负载时，这些参数应取自同一条记录。

| 用途 | 入口或配置 |
| --- | --- |
| 运行智能体 | `main.py`。用 `.venv-agent/bin/python main.py --help` 查看完整参数 |
| 初始硬件和搜索空间 | `configs/initial_hardware.json`、`configs/paper_initial_hardware/`、`configs/search_space.json` |
| 切换智能体或历史检索 | 默认多智能体分层搜索。用 `--agent-mode single` 运行单智能体对照，用 `--no-rag` 关闭案例检索与写入 |
| 控制每次提案的编辑数 | `--max-edits 1` 限制为一次编辑。开展原子编辑对比时显式设置 |
| 五种传统基线 | `rs` 随机搜索、`sa` 模拟退火、`bo` 贝叶斯优化、`ga` 遗传算法、`sweep` 参数扫描，见 [基线说明](baselines/README.md) |
| 批量论文实验 | `scripts/run_paper_agents.py` 和 `scripts/run_paper_baselines.py`。先用 `--help` 检查负载、预算和输出路径 |

例如，在 ViT-Tiny 上运行随机搜索，从同一初始设计评估一个候选：

```bash
.venv-agent/bin/python -m baselines.runner --algo rs --workload vit_tiny \
    --nn-model-factory tests.factory:build_vit_tiny --budget 1 --seeds 0 \
    --output-root runs/smoke-baseline
```

## 运行产物

每次智能体运行会在 `--output-root` 下创建 `run_iteration_NNN/`。轮次 0 单独记录初始设计，后续编号对应搜索轮次。在输出中可以核对模型建议了什么、调度器选择了什么模式，以及仿真器实际评估了什么。

| 位置 | 内容 |
| --- | --- |
| `iter_000/` | 初始硬件 `hardware.json`、工作负载 `workload.json`、综合目标和分项指标 `metrics.json`、仿真报告 `report.json` |
| `iter_001/` 及后续轮次 | 该轮实际候选配置、仿真指标、能耗和面积明细 |
| `iter_NNN/analysis/` | 调度决定 `schedule.json`、各智能体的结构化结论、模型调用轨迹 `llm_trace.jsonl` |
| `summary.json`、`trajectory.json` | 运行汇总及逐轮轨迹，后者可通过 `--no-trajectory` 关闭 |

若输出中保留某轮的分析记录，但尚无该轮的 `metrics.json`，应先检查该轮是否完成仿真。搜索效率统计可以分别使用当轮候选目标与截至该轮的历史最优目标。

## 测试与迁移

```bash
.venv-agent/bin/python tests/run_tests.py
```

换机器时从源码克隆，在目标机执行安装脚本，再运行“快速开始”中的单轮命令。Agent 与仿真器分别使用各自的 Python 环境，通过 MCP 子进程通信。GitHub Actions 的测试配置包含 Agent 单元测试，以及在全新 Ubuntu 环境中编译 CACTI、运行一次真实 SimPhony 仿真的任务，见 [ci.yml](.github/workflows/ci.yml)。

源码仓库包含环境配置、依赖清单、工作负载描述、模型工厂和仿真器源码。本机密钥 `.env`、环境目录 `.venv-*/`、工具缓存 `.tools/`、运行目录 `runs/` 和实验归档 `experiment/` 由 [.gitignore](.gitignore) 排除。迁移历史实验时另行复制所需的运行记录；新安装可直接生成新的记录。

## 代码导览

| 目录 | 内容 |
| --- | --- |
| `photon_tuner/agents/`、`photon_tuner/runtime/` | 智能体定义、工作流、调度和逐轮搜索 |
| `photon_tuner/toolbox/`、`photon_tuner/skills/` | 可由智能体调用的工具与组合分析操作 |
| `photon_tuner/rag/` | 历史案例存储与检索 |
| `photon_tuner/bridge/mcp/`、`photon_tuner/simulators/` | 仿真器通信、适配器和 SimPhony 源码 |
| `baselines/`、`scripts/`、`tests/` | 传统搜索、安装及实验入口、测试和负载工厂 |

搜索空间定义见 [search_space.json](configs/search_space.json)，指标和单位见 [units.md](docs/design/units.md)。

## 第三方代码

[`photon_tuner/simulators/simphony/`](photon_tuner/simulators/simphony) 是
[ScopeX-ASU/SimPhony](https://github.com/ScopeX-ASU/SimPhony) 的 vendored 副本
（MIT 许可证；论文 arXiv:2411.13715）。详见其附带的
[`LICENSE`](photon_tuner/simulators/simphony/LICENSE) 与
[`VENDORED.md`](photon_tuner/simulators/simphony/VENDORED.md)。其中的 CACTI 子组件
来自 Hewlett-Packard（见 `cacti/` 下的头文件）。

真实 SimPhony 栈固定在 Python 3.10 / torch 1.12，因此通过 MCP 桥进程外运行，
不与主运行时共享解释器。
