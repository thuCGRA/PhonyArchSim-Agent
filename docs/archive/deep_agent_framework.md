# photonic-tuner：deep agent 框架实现说明

本文说明本项目如何用 **deepagents（LangChain）** 框架重写，覆盖总体架构、五个
subagent 流水线、结构化输出与证据门禁的工程决策、工具层、以及 real 模式的 MCP
进程外桥接。目标读者是后续要维护/扩展本项目的开发者。

---

## 1. 总体架构

```
                    ┌──────────────────────────────────────────────┐
                    │            DeepAgentTuner（编排宿主）          │
                    │   photon_tuner/runtime/orchestrator.py        │
                    │  负责：顺序调度 / keep-best / forbidden 指纹   │
                    │        repair-duplicate / run_dir / trajectory │
                    └──────────────┬───────────────────────────────┘
                                   │ 确定性顺序（Python 宿主决定，非 LLM）
        ┌──────────────┬───────────┼───────────┬──────────────┐
        ▼              ▼           ▼           ▼              ▼
   model agent  architecture  topology    device        solution
        │              │           │           │              │
        └────── 都通过 create_deep_agent(model, tools, prompt, middleware) 构建
                                   │
                    ┌──────────────┴──────────────┐
                    │         Evidence 门禁        │
                    │   8 行协议：确定性 1/4/5     │
                    │   + Paratera judge 2/3/6/7  │
                    │   + 规则决策 8              │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │        SimulatorAdapter      │
                    │   analytic（进程内镜像）      │
                    │   real    （MCP 进程外桥）    │
                    └──────────────┬──────────────┘
                                   ▼
                     objective / metrics（core 领域逻辑，未改动）
```

五个 agent 的顺序是**在 Python 宿主里硬编码的**：`model → architecture → topology →
device → solution`。deepagents 的编排图（graph）本身会让模型自主决定工具调用，
但"哪个 agent 什么时候跑"这一层不交给 LLM，从而保持 DSE 的确定性。

---

## 2. 文件地图（新代码）

| 路径 | 职责 |
|---|---|
| `photon_tuner/runtime/orchestrator.py` | `DeepAgentTuner`：编排宿主，串 5 个 agent + 门禁 + 仿真 |
| `photon_tuner/agents/builders.py` | 5 个 agent 工厂（`build_model_agent` 等） |
| `photon_tuner/agents/prompts.py` | 5 个角色 system prompt |
| `photon_tuner/agents/outputs.py` | pydantic 结构化输出模型（`ModelOutput` 等） |
| `photon_tuner/agents/evaluate.py` | 8 行证据门禁（确定性 + judge + 决策） |
| `photon_tuner/agents/evidence.py` | 离线 EvidenceAnalystAgent（探针交叉分析） |
| `photon_tuner/tools/__init__.py` | 工具包装层：`RunState` / `json_schema_to_pydantic` / `wrap_registry` |
| `photon_tuner/tools/topology_tools.py` | `summarize_topology` / `inspect_light_path` |
| `photon_tuner/core/topology_spec.py` | `TopologyProvider` 协议 + `StaticTopologyProvider` 占位 |
| `photon_tuner/llm/model.py` | `build_chat_model`（Paratera / DeepSeek） |
| `photon_tuner/deep/mcp/simphony_server.py` | real 模式 MCP 服务端（Python 3.10） |
| `photon_tuner/deep/mcp/simphony_tool.py` | `SimphonyMCPClient` 客户端（deepagents 侧） |
| `photon_tuner/testing/fake_model.py` | 离线测试用 `FakeToolCallingModel` |
| `tests/deep/` | 13 个离线测试（fake 模型驱动，无需网络/LLM） |
| `scripts/dev/smoke_deep_tune.py` | analytic 真实 Paratera 端到端冒烟 |
| `scripts/dev/smoke_mcp.py` | real 模式 MCP 往返冒烟 |

**保留不动的领域层**：`core/`（schemas / search_space / objective / workload）、
`simulators/`（analytic 镜像 + simphony adapter + vendored SimPhony）、`toolbox/`
（19 个领域 handler）、`rag/`、`evidence/`。只有 `common/tool_types.py` 被工具层复用，
其余领域逻辑零改动。

---

## 3. 五个 subagent 各自做什么

| Agent | 输入 | 工具面 | 输出（pydantic） |
|---|---|---|---|
| **model** | 硬件摘要 + metrics + 工作负载 | MODEL_TOOLS（9） | `ModelOutput`：全局瓶颈层候选 |
| **architecture** | model 的候选 + 逐层诊断数据 | ARCHITECTURE_TOOLS（7） | `ArchitectureOutput`：`root_cause_summary` + `layer_diagnoses` |
| **topology** | 光路损耗/功率预算 | TOPOLOGY_TOOLS（7，含 `summarize_topology`/`inspect_light_path`） | `TopologyOutput`：`loss_budget_diagnosis` / `dominant_loss_contributors` / `power_budget_margin` |
| **device** | 器件库 + 约束 | DEVICE_TOOLS（5） | `DeviceOutput`：`recommended_link_edits` + `retrieval_description` |
| **solution** | 全部上游结论 + 摘要（只读） | SOLUTION_TOOLS（3：`summarize_hardware_config`/`summarize_metrics`/`compare_search_states`） | `SolutionOutput`：`strategy` / `actions` / `link_edits` / `rationale` / `notes` |

- 每个 agent 通过 `create_deep_agent(model, tools, system_prompt, backend,
  middleware=[FilesystemMiddleware(tools=["read_file"])], name=...)` 构建。
- **solution agent 是只读的**：它只**提案** `actions`（架构参数编辑）和 `link_edits`
  （器件参数编辑）；宿主用 `EDIT_TOOLS={modify_hardware_parameter, step_hardware_parameter}`
  和 `LINK_EDIT_TOOLS={modify_device_parameter}` **确定性、逐条容错**地落地。

---

## 4. 关键工程决策

### 4.1 两阶段结构化输出（替代 response_format）

deepagents 默认可用 `response_format`（ToolStrategy）让模型直接输出结构化结果，但
在 Paratera 的**推理模型**（`DeepSeek-V4-Pro-0813`）上**多轮不稳定**——模型偶发不调用
respond 工具，导致空输出。因此改为两阶段：

1. **阶段一**：agent **不带** `response_format`，自由调用领域工具收集证据，最后写
   **纯文本**结论（prompt 尾部附有字段清单，见 `builders.py::_field_spec`）。
2. **阶段二**：宿主从 transcript 里取最后一段 AI 文本，用
   `model.with_structured_output(OutputModel)` **单轮强转**为 pydantic 对象
   （`orchestrator.py::_invoke`）。该路径在 Paratera 上已单独验证可靠。

### 4.2 收敛工具面 + 剥文件系统工具

deepagents 默认注入 `ls/glob/grep/read_file/write_file/edit_file/execute` 等文件系统
工具。实测 solution agent 会用它"漫游"运行目录（几十条 ls/glob/read），撞
`recursion_limit` 却不回答。修复：

- 所有角色 agent 的 `FilesystemMiddleware` 换成 `tools=["read_file"]`（`read_file`
  是 FilesystemMiddleware 强制保留项，不能给空列表）。
- `_invoke` 显式传 `config={"recursion_limit": 40}`。
- solution 的工具面收敛到 3 个只读摘要工具。

### 4.3 工具包装层（None 值剥离）

`json_schema_to_pydantic` 会把每个字段变成 `Optional[...] = None`，LangChain 的
`StructuredTool` 对未传参数填 `None`，导致下游 handler 的
`arguments.get(key, default)` 拿到 `None` 而非默认值（曾触发 `int(None)` 崩溃）。
修复：`tools/__init__.py::_call` 在委托 handler 前**剥离 None 值**。

### 4.4 max_tokens 预算

推理模型会把 `max_tokens` 全花在 reasoning 上（曾出现 `reasoning_tokens=4096,
text_tokens=None`，`LengthFinishReasonError`）。`configs/llm.json` 的 `max_tokens`
已提到 **16384**。

---

## 5. 编排不变量（从旧 tuner 保留）

- **keep-best**：只接受 objective 严格下降的候选。
- **forbidden 指纹**：`hardware_fingerprint` 记入集合，避免重复评估同一点。
- **`_repair_duplicate`**：应用编辑后若与 forbidden 撞车则回退修复。
- **打包 run_dir**：`run_iteration_NNN/` + `analysis/{model_level,architecture,
  topology,device,solution_proposal,gate_report}.json` + `trajectory.json`。
- **objective 对齐**：`_evaluate` 逐字等价于旧 `tuner._evaluate`
  （`design_space.normalize → adapter.evaluate → metrics_with_objective`），
  由 `tests/deep/test_objective_alignment.py` 锁定。

---

## 6. Evidence 门禁（8 行协议）

`agents/evaluate.py`，每轮 solution 之后跑，产物写入 `gate_report.json`：

| 行 | 名称 | 判定方式 |
|---|---|---|
| 1 | 格式（schema 合规） | 确定性（pydantic 校验） |
| 2 | 完整性 | Paratera judge |
| 3 | 证据完整性 | Paratera judge |
| 4 | 证据真实性（来源可查） | 确定性（无工具 transcript → 不通过） |
| 5 | 一致性（声明编辑 vs 实际应用） | 确定性（对比 `applied/failed`） |
| 6 | 正确性 | Paratera judge |
| 7 | 适用性 | Paratera judge |
| 8 | 接受/反馈决策 | 规则：硬行(1/5)不过→重执行；行4不过→请求补充信息；全过→接受 |

judge 用 `model.with_structured_output(JudgeVerdict)` **单次调用**完成（不套
create_deep_agent），失败时门禁降级为"接受"并记录原因，绝不中断主循环。

---

## 7. Topology 占位与 search-space 预留

- `core/topology_spec.py` 定义 `TopologyProvider` 协议（`graph()` / `optical_path()`）。
- 当前用 `StaticTopologyProvider`：光路 `laser → splitter → waveguide → mzm →
  mrr_rerouter → coupler → photodetector`，`_NON_DEVICE_KEYS` 排除激光源/分束器等非
  器件节点。
- 未来接入真 SimPhony 拓扑时，实现一个 `SimphonyTopologyProvider` 即可，工具层
  （`summarize_topology`/`inspect_light_path`）与 agent 层无需改动。

---

## 8. real 模式：MCP 进程外桥接

### 8.1 为什么需要进程外

| 依赖 | Python 要求 |
|---|---|
| deepagents 0.7.29 | `>=3.11` |
| torch 1.12.1（真 SimPhony 依赖） | 仅 cp37–cp310，**无 cp311+ wheel** |

两者无法共用一个解释器，因此真 SimPhony 在 Python 3.10 venv（`.venv-real`）里跑，
deepagents 在 3.14 venv（`.venv-deep`）里跑，中间用 MCP over stdio 桥接。

### 8.2 数据流

```
cli.py --mode real (3.14)
  → SimphonyAdapter(transport="mcp").evaluate(hardware)
  → SimphonyMCPClient（后台事件循环持有常驻 stdio 会话）
  → subprocess: .venv-real/bin/python simphony_server.py (3.10)
  → FastMCP 工具 evaluate(hardware) → SimphonyAdapter(mode="real") → ONNArchSimulator
  → 结果 JSON 回传 → 3.14 侧算 objective → 继续流水线
```

### 8.3 两个关键细节

1. **stdout 隔离**：真 SimPhony 和 CACTI 子进程会把进度打印到 stdout，会污染 MCP
   的 stdio JSON-RPC 流（客户端把每行 stdout 都当协议消息解析）。服务端在 tool
   处理器内用 `os.dup2` 把 fd1 重定向到 fd2（stderr），覆盖 Python `print` 与子进程输出。
2. **返回解包**：`langchain-mcp-adapters` 的工具返回内容块
   `[{"type":"text","text":"..."}]`，客户端 `_unwrap_text` 正确解出 JSON 字符串。

### 8.4 验证

`scripts/dev/smoke_mcp.py` 通过 MCP 评估一个 vit_tiny 点，指标与直连 real 模式
**逐位一致**：`objective=10204.406310093758`、`energy_pj=216580750.2660602`、
`area_um2=66501954.24`、`latency_s=0.0007882927829770002`。

> 注意：analytic 镜像的 vit_tiny 基线 objective 是 `73137.08`，与 real 的 `10204.41`
> 不同——**这是预期行为**：analytic 是近似镜像，数值上不等于真 SimPhony。"objective
> 对齐"指的是*代码路径*对齐（deep 流水线 == 旧 tuner 的同一套 simulator+objective），
> 而非 analytic == real。

---

## 9. 如何运行

```bash
# 依赖环境（已就绪）
#   .venv-deep   Python 3.14  deepagents / langchain-mcp-adapters
#   .venv-real   Python 3.10  torch 1.12.1 / mcp 1.30（真 SimPhony）

# analytic 模式（进程内镜像，最快）
.venv-deep/bin/python -m photon_tuner.cli --mode analytic --workload vit_tiny --iterations 3

# real 模式（MCP 进程外，真 SimPhony）
.venv-deep/bin/python -m photon_tuner.cli --mode real --workload vit_tiny \
    --nn-model-factory tests.factory:build_vit_tiny --iterations 1

# 冒烟脚本
.venv-deep/bin/python scripts/dev/smoke_deep_tune.py   # analytic + 真实 Paratera
.venv-deep/bin/python scripts/dev/smoke_mcp.py         # real + MCP 往返

# 测试
.venv-deep/bin/python tests/deep/run_deep_tests.py     # 13 个离线 deep 测试
python3 tests/run_tests.py                              # 94 个领域测试
```

---

## 10. 扩展指南

- **新增一个 agent**：在 `agents/prompts.py` 写 role prompt → `agents/outputs.py` 写
  pydantic 模型 → `agents/builders.py` 写 `build_xxx_agent` → 在
  `orchestrator.tune()` 的序列里插入一次 `_invoke`。
- **新增领域工具**：复用 `tools/__init__.py::wrap_registry`（把 toolbox handler 包装成
  LangChain tool）或 `json_schema_to_pydantic`（从 JSON schema 生成参数模型）。
- **替换拓扑数据源**：实现 `TopologyProvider` 协议并在 `_build_agents` 传入新 provider。
- **去掉 MCP 依赖（可选）**：若未来把真 SimPhony 的 torch 升级到支持 3.11+，可直接
  `transport="inprocess"`，删除 `deep/mcp/` 即可，agent 层零改动。
