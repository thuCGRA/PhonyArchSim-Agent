# 测试（tests/）

本目录是项目的**统一测试目录**，集中放置所有单元/集成测试。测试基于 Python 标准库
`unittest`，无第三方测试框架依赖。

- `tests/*.py` —— **领域测试**（纯标准库 + PyYAML，无第三方依赖）。
- `tests/deep/` —— **deep-agent 运行时测试**，需要 `langgraph`（在 `.venv-agent` 下运行）。

## 如何运行

在项目根目录执行：

```bash
# 领域 + deep 全部测试（推荐用 .venv-agent，包含 langgraph）
.venv-agent/bin/python tests/run_tests.py

# 仅领域测试（普通 python3 即可；缺 langgraph 时会自动跳过 tests/deep/ 并提示）
python3 tests/run_tests.py
```

`tests/run_tests.py` 会先加载顶层 `test_*.py`（领域），再尝试 `import langgraph`：
可导入则追加 `tests/deep/`，不可导入则打印提示并跳过。当前全量 372 项（领域 254 + deep 118）。

> 测试输入一律来自 `tests/fixtures/`（见 `tests/fixtures/README.md`），**不要**读取
> `runs/` —— 该目录被 `.gitignore` 排除，CI 全新检出时并不存在。

CI 中的调用方式（见 `.github/workflows/ci.yml`）：

```bash
coverage run --source=photon_tuner tests/run_tests.py
```

## 目录结构

```
tests/
├── run_tests.py            # 统一测试入口
├── __init__.py
├── factory.py              # real 模式 CPU 模型工厂（对外路径 tests.factory:build_<workload>）
├── cnn_models.py           # CNN 模型实现（QVGG），被 factory.py 引用
├── smoke_real.py           # real 模式手动冒烟脚本（非单元测试）
├── fixtures/               # 随仓库提交的测试基线（见 fixtures/README.md）
├── test_*.py               # 领域测试（见下表）
└── deep/                   # deep-agent 运行时测试（需要 langgraph）
    ├── __init__.py
    └── test_*.py           # 见下表
```

## 领域测试（tests/*.py）

| 文件 | 说明 |
|---|---|
| `test_analytic.py` | 对照 SimPhony 参考值校验离线 analytic 镜像 |
| `test_cache.py` | `photon_tuner.llm.cache`（进程内响应缓存） |
| `test_design_space.py` | 分层/多粒度/自适应设计空间 |
| `test_device_frontier.py` | 真实变体 Pareto 前沿摘要 |
| `test_evidence.py` | 搜索空间有效性验证证据层 |
| `test_mesh_eval.py` | analytic MZI-mesh 求值器 |
| `test_mesh_model.py` | MZI-mesh 拓扑模型（MZI 数量/深度） |
| `test_objective.py` | 光子目标函数与插入损耗预算惩罚 |
| `test_pareto.py` | 多目标 Pareto 归档（吞吐归一化指标） |
| `test_prompt_golden.py` | 角色提示词 golden 回归 |
| `test_provenance.py` | 器件出处审计（cfgs 数值可追溯到论文或被标记） |
| `test_search_space.py` | 搜索空间步进、钳位与器件编辑 |
| `test_simphony.py` | real 模式硬件注入桥 |
| `test_tool_cache.py` | `photon_tuner.toolbox.tool_cache`（report 工具结果缓存） |
| `test_truncate.py` | `photon_tuner.common.truncate`（上下文长度预算） |
| `test_weight_array_eval.py` | WDM 权重阵列求值器（MRR 权重组 / PCM 交叉开关 / 单 WDM 链路） |
| `test_workload.py` | 可加载工作负载层（analytic 模式下的真实模型） |

## deep-agent 测试（tests/deep/）

> 依赖 `langgraph`，需要 `.venv-agent/bin/python`；内部使用 `FakeToolCallingModel`，离线、确定性、无需网络/LLM。

| 文件 | 说明 |
|---|---|
| `test_deep_pipeline.py` | LangGraph DSE 流水线离线端到端（假模型） |
| `test_evaluate.py` | 各阶段证据评估门控（确定性行为 + 决策） |
| `test_mcp_bridge.py` | MCP 主机侧明细文件物化的离线测试 |
| `test_mesh_end_to_end.py` | 通过智能体流水线的端到端网格搜索 |
| `test_objective_alignment.py` | 与共享遗留域路径的目标对齐验证 |
| `test_sequential_joint.py` | 顺序搜索与联合搜索的可消融对比 |
| `test_state_snapshot.py` | 检查点快照契约 |
| `test_topology.py` | 电路拓扑提供器与工具 |
| `test_topology_space.py` | 拓扑搜索空间（模板注册表 + 耦合矩阵） |
| `test_vector_store_features.py` | RAG 存储新特性 |

## 辅助文件（非测试）

- `factory.py` —— real 模式模型工厂，对外路径为 `tests.factory:build_<workload>`
  （如 `build_vit_tiny`），被 CLI（`--nn-model-factory`）、MCP 桥（`simphony_server.py` 默认值）
  和 `scripts/` 实验脚本引用，**不要改动其模块路径或删除**。
- `cnn_models.py` —— CNN 模型实现（`QVGG`），被 `factory.py` 引用。
- `smoke_real.py` —— real 模式手动冒烟脚本，需要 `.venv-simulator`（torchonn 等），
  不属于自动测试，不参与 `run_tests.py`。

## 编写约定

- **命名**：文件 `test_*.py`，类继承 `unittest.TestCase`；自动发现按此约定匹配。
- **放置位置**：不依赖 `langgraph` 的放 `tests/` 顶层；依赖 deep-agent 运行时的放 `tests/deep/`。
- **项目根路径**：用 `Path(__file__).resolve().parents[N]` 计算——
  `tests/test_xxx.py` 用 `parents[1]`；`tests/deep/test_xxx.py` 用 `parents[2]`（多一层）。
- **无需网络的 real 依赖**：优先用 `factory.py` 提供的 CPU 模型 / `FakeToolCallingModel`，保证测试离线可复现。
- 若新增 deep 测试依赖了新的第三方包，请在 `requirements.txt` 注明并在 README 说明运行环境。
