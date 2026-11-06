# Deep Agents 迁移实施方案（精确到文件）

> 状态：已全部实施完成（M0–M5）。analytic 端到端跑通并经真实 Paratera 冒烟验证；
> M5 MCP real 模式已实现并验证往返指标与直连 real 模式逐位一致。详见
> `docs/design/deep_agent_framework.md`。

## 0. 范围与原则

- **新增 `photon_tuner/deep/` 包**承载所有 deepagents 集成。
- **旧代码一律不动**：`photon_tuner/runtime/`、`photon_tuner/agents/`、`photon_tuner/llm/clients.py`、
  以及 `cli.py` 的 legacy 路径，保持 120 个测试全绿。
- **复用不重写**：`core/`（schemas、search_space、objective、workload、topology）、`rag/`、
  `simulators/analytic.py`、`simulators/base.py`、`toolbox/`（工具 handler）全部 import 复用。
  只有「LLM 协议、agent 编排、工具包装」是新的。
- 已锁定决策：5 agent（model→architecture→topology→device→solution）；损耗/功率预算归 topology；
  evaluate 用 Paratera judge（不用 LangSmith）；topology 先占位+预留接口；RAG/memory 暂缓；
  第一阶段 analytic 进程内 import，第二阶段 real 走 MCP。

## 1. Phase 0 — 环境

```bash
cd /home/fyq/epic
export UV_CACHE_DIR=/home/fyq/epic/.tools/cache   # 沙箱下 ~/.cache/uv 不可写
.tools/uv venv .venv-deep --python 3.14
.venv-deep/bin/pip install deepagents langchain-openai pyyaml
```

- 新增 `requirements-deep.txt`（3.12/3.14 侧依赖）。
- `.venv-real`（3.10，torch 1.12.1）保持不动，第二阶段 MCP 才用到。

## 2. Phase 1 — LLM 模型适配

**新增 `photon_tuner/deep/model.py`**

- `build_chat_model(cfg) -> ChatOpenAI`：
  `base_url=https://llmapi.paratera.com/v1`，`model=DeepSeek-V4-Pro-0813`，
  `api_key=resolve_api_key(cfg)`（复用 `photon_tuner/llm/clients.py:277`），
  `temperature/max_tokens/timeout` 来自 `llm/config.py` 的 `LLMConfig`。
- 结构化输出：`model.with_structured_output(PydanticModel)`，替代
  `runtime/llm_completion.py:complete_json_continuing_messages` + `validate_response`。

## 3. Phase 2 — 工具层

**新增 `photon_tuner/deep/tools.py`**

- 通用适配器：遍历 `AnalysisToolbox`/`DeviceToolbox` 的 `name -> (ToolSpec, handler)` 注册表
  （`toolbox/analysis_tools/__init__.py:24`），把 `ToolSpec.input_schema`（JSON Schema）转 pydantic，
  包成 `StructuredTool`。
- 状态注入：每次 run 实例化一个 `ToolContext`（复用 `toolbox/common_tool/context.py:17`）并闭包捕获。
  3 个 mutation 工具（`modify_hardware_parameter`/`step_hardware_parameter`/`modify_device_parameter`）
  就地改 `context.current_hardware` 的语义不变。
- 开启 `FilesystemMiddleware`（root=run_dir），agent 直接 ls/read_file/grep `iter_NNN/*.json`；
  16 个只读 summary 工具保留。

## 4. Phase 3 — Agent 层（5 个 subagent）

**新增 `photon_tuner/deep/agents/`**

| subagent | prompt 来源 | 工具 | 输出（pydantic） |
|-|-|-|-|
| `model_agent` | `MODEL_LEVEL_PROMPT` | 9 只读 + filesystem | `ModelAnalysisResult` |
| `architecture_agent` | `LAYER_LEVEL_PROMPT`（改架构分析） | 7 layer 工具 + filesystem | `BottleneckState` |
| `topology_agent` | 新写 | 光学结构只读 + `summarize_topology` + filesystem | `TopologyAnalysisResult`（损耗/功率预算） |
| `device_agent` | `OPTICAL_LEVEL_PROMPT`（只器件选择+参数） | device catalog 工具 | `DeviceSelectionResult`（link edits） |
| `solution_agent` | `SOLUTION_PROMPT` | 只读 + 3 mutation | `SolutionProposal` |

拆分依据：现有 `agents/optical_level.py` 的 `loss_budget_diagnosis`+`power_budget_margin` → topology；
`recommended_link_edits` → device。

## 5. Phase 4 — 编排层

**新增 `photon_tuner/deep/orchestrator.py`**

- `DeepAgentTuner.tune()`：近移植 `runtime/tuner.py:107`，顺序
  `model→architecture→topology→device→solution`。
- 保留 `_is_better`（keep-best）、`forbidden` + `_repair_duplicate`、`_next_run_dir`、
  `trajectory.json`。这是 results.md `LLM vs fallback` 标签能继续工作的前提。

## 6. Phase 5 — Evidence evaluation 门禁

**新增 `photon_tuner/deep/evaluate.py`**

- 确定性校验（第 1/4/5 行）：pydantic schema、证据来源可追溯（LangGraph state 的 tool call 出处）、
  声称修改 vs 真实执行 diff（复用 `applied_edits`/`failed_edits`）。
- LLM 语义校验（第 2/3/6/7 行）：`build_chat_model`（Paratera judge），规则+LLM。
- 验收决策（第 8 行）：接受/补信息/重验/重执行/升级规划，规则主导。
- 挂点：逐级只跑确定性校验；末端 solution 后跑完整 8 项。
- 实现为 LangChain `AgentMiddleware`（可 interrupt/reject）。

## 7. Phase 6 — Topology 占位 + 预留接口

**新增 `photon_tuner/core/topology_spec.py`**

- `TopologyProvider` 协议：`topology(hardware) -> TopologyGraph`。
- `StaticTopologyProvider`：从 hardware + 器件目录推导默认拓扑（占位）。
- 预留 `SimphonyTopologyProvider` 供未来真实扩展。
- 新增只读工具 `summarize_topology` / `inspect_light_path`。
- 现有 `LAYERED_DESIGN_SPACE`（system/optical_core/device/device_param）不含 topology 层，
  拓扑结构作为独立 schema 预留。

## 8. Phase 7 — SimPhony 进程外（MCP，第二阶段）✅ 已实现

- **`photon_tuner/deep/mcp/simphony_server.py`**（3.10）：MCP 服务，暴露 `evaluate(hardware)`，
  复用 `SimphonyAdapter(mode="real")`；tool 处理器内把 fd1 重定向到 fd2，避免 SimPhony/CACTI
  的 stdout 污染 stdio JSON-RPC 流。
- **`photon_tuner/deep/mcp/simphony_tool.py`**（deepagents 侧）：`SimphonyMCPClient` 客户端，
  同步 `evaluate` 签名同进程内 adapter，后台事件循环持有常驻 stdio 会话。
- 因 `simulators/base.py SimulatorAdapter` 已是抽象，替换只动 `SimphonyAdapter(transport="mcp")` 一处。
- 冒烟 `scripts/dev/smoke_mcp.py`：往返指标与直连 real 模式逐位一致（objective=10204.406310093758）。

## 9. Phase 8 — CLI / 配置

- `cli.py` 加 `--framework {legacy,deep}`（默认 legacy）；或新增 `main_deep.py` 入口。
- `configs/llm.json` 不动，deep 侧共用 Paratera key。

## 10. Phase 9 — 测试与不变量

- 旧 120 测试保持全绿。
- 新增 `tests/deep/`：工具包装、structured output、门禁确定性行、offline orchestrator。
- 离线冒烟：自定义假 tool-calling `BaseChatModel`（~40 行）。
- 必须保持：run_dir 布局、`summary["iterations"]`/`best`、trajectory.json、trace 格式。

## 11. 里程碑

| M | 内容 | 依赖 | 预估 |
|-|-|-|-|
| M0 | venv + build_chat_model + trivial tool 跑通真实 Paratera | Phase 0/1 | 小 |
| M1 | 工具适配器 + 5 subagent + orchestrator（analytic 端到端） | M0 | 中 |
| M2 | 确定性门禁 + 末端完整 8 项（Paratera judge） | M1 | 中 |
| M3 | topology 占位 + 预留接口 | M1 | 中 |
| M4 | tests/deep + 与 legacy 同 workload 对比 objective（对齐） | M2/M3 | 中 |
| M5 | MCP real 模式 | M4 后按需 | ✅ 完成（`smoke_mcp.py` 验证） |

风险集中在 M4：新框架同一 analytic workload 必须得到与 legacy 一致的 baseline objective。

## 12. 回归基线

- 执行前记录：`python3 tests/run_tests.py`（120 全绿，0.68s）、
  legacy 的 vit_tiny/vit_base baseline objective（见 `docs/results/results.md`）。
- 迁移后逐项对齐。
