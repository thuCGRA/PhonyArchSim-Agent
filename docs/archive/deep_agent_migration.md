# 迁到 Deep Agents 框架：可行性评估与改动清单

> 结论先行：\\\*\\\*技术上可行，模型侧已验证具备条件；但真正的卡点不是 agent 代码，而是 Python 解释器。\\\*\\\*
> Deep Agents（LangChain `deepagents` 包）要求 Python ≥ 3.11，而 real 模式的 SimPhony 栈被
> `torch==1.12.1+cpu` 钉死在 Python 3.10。因此只有两条路：拆成双解释器/双进程，或者
> 只移植 Deep Agents 的\\\*\\\*架构模式\\\*\\\*而不引入 langchain 依赖。

评估基于当前工作区（`v1` 分支，commit `c4ea44e`）的实际代码与实测数据，不是纸面推演。

\---

## 1\. 现状盘点（已核实）

### 1.1 规模

|部分|行数|
|-|-|
|`photon\\\_tuner/` 自有代码（不含 vendored SimPhony）|8,557|
|`photon\\\_tuner/simulators/simphony/`（vendored，MIT）|7,715|
|`tests/`|2,175|
|`scripts/`|791|

* 自有代码里 **agent 运行时只有约 600 行**（`runtime/react\\\_loop.py` 183 + `runtime/tuner.py` 428

  * `runtime/llm\\\_completion.py` 71 + `runtime/results.py`）；agent 角色 4 个类合计约 640 行。
* 测试：**120 个用例全绿，0.68 s**（`python3 tests/run\\\_tests.py`，系统 Python 3.14.4）。

### 1.2 架构（as-is）

```
main.py → photon\\\_tuner/cli.py
        → runtime/tuner.py:107 MultiAgentTuner.tune()          ← 硬编码的确定性 4 段流水线
             for iteration in 1..N:
               model\\\_level.analyze()    ┐
               layer\\\_level.analyze()    │ 每个角色 = 一次 ReAct 工具循环（max\\\_steps=8）
               optical\\\_level.analyze()  │          + 一次「最终严格 JSON」补全
               solution.propose()       ┘
               \\\_evaluate() → adapter.evaluate()  ← SimPhony（analytic 镜像 / real torch 栈）
               keep-best / forbidden fingerprint / \\\_repair\\\_duplicate 兜底
```

四个关键实现选择：

1. **LLM 协议是自定义的，不是 tool calling。** `llm/clients.py:201 OpenAICompatibleClient`
用手写 `urllib` 调 `/chat/completions`，只取 `choices\\\[0].message.content`
（`clients.py:250`），然后 `extract\\\_json\\\_object()`（`clients.py:114`）从文本里抠一个
JSON 对象。**响应里的 `tool\\\_calls` 被完全丢弃**——工具调用是靠 prompt 里约定
`{"action":"tool","tool":{...}}` 这种 JSON 形状模拟的（`llm/prompts.py:11`）。
2. **工具是带可变运行态的注册表，不是无状态函数。** 19 个工具
（`analysis\\\_tools` 12 个 + `device\\\_tools` 7 个）通过 `name -> (ToolSpec, handler)`
注册（`toolbox/analysis\\\_tools/\\\_\\\_init\\\_\\\_.py:24`），每个 handler 都收一个
`ToolContext`（`toolbox/common\\\_tool/context.py:17`）。其中 3 个是**原地修改**语义：
`modify\\\_hardware\\\_parameter` / `step\\\_hardware\\\_parameter` / `modify\\\_device\\\_parameter`
直接改写 `context.current\\\_hardware`，再由 `agents/solution.py:211` 读回。
其余 16 个是只读证据读取，靠 `context.output\\\_dir` 去磁盘上读
`runs/.../iter\\\_NNN/\\\*.json`。
3. **角色之间不通过消息通信，通过磁盘产物 + 类型化 dataclass。**
每轮产出 `iter\\\_NNN/{report,energy\\\_detail,area\\\_detail,insertion\\\_loss,workload,memory}.json`
和 `iter\\\_NNN/analysis/{model\\\_level,layer\\\_level,optical\\\_level}\\\_result.json`

   * `solution\\\_proposal.json`，下游 agent 拿到的是 `ModelAnalysisResult` /
`BottleneckState` / `OpticalAnalysisResult` 这些 dataclass（`core/schemas.py`）。
4. **有个第 5 个 agent 不在主循环里。** `agents/evidence.py:46 EvidenceAnalystAgent`
离线读取确定性消融证据（`evidence/probe.py`），产出 `verified\\\_levers` /
`null\\\_results` / `anomalies` / `search\\\_policy`。它是 subagent 的天然候选。

### 1.3 量化基线（vit\_base，5 轮真实运行，`runs/legacy/convergence\\\_vit\\\_base/trace/`）

|指标|数值|
|-|-|
|LLM 调用总数|117（`react\\\_tool\\\_use` 97 + 4 个最终分析各 5）|
|每轮调用数|≈ 23.4（≈4.85 次工具步/角色，上限 8）|
|每轮 prompt tokens|≈ 96,159|
|每轮 completion tokens|≈ 41,978（其中 reasoning tokens 占大头，单次样本 4628/4988）|

这组数字是后面评估「Deep Agents 会不会更贵」的对照基准。

\---

## 2\. Deep Agents 是什么，和本项目怎么对应

`deepagents`（当前 0.7.28）是 LangChain 在 LangGraph 之上的一层**opinionated agent harness**：
`create\\\_deep\\\_agent(model, tools, system\\\_prompt, subagents, middleware, backend, store, ...)`。
它内置的能力与本项目现状的对应关系：

|Deep Agents 内置能力|本项目现有对应物|迁移判断|
|-|-|-|
|`subagents=\\\[...]` 独立上下文窗口|4 个 agent 类，各自独立消息栈|**天然对齐**，几乎是 1:1|
|每 subagent 独立 `tools=\\\[...]`|`ALLOWED\\\_TOOLS` / `READ\\\_ONLY\\\_TOOLS` 白名单（`agents/\\\*.py:19/32`）|直接可搬|
|内置 `write\\\_todos` 规划工具|`tuner.py:155` 硬编码的 `for iteration` + 4 段顺序|⚠️ 会丢掉「每轮 4 个角色必跑」的保证|
|虚拟文件系统（ls/read\_file/write\_file/glob/grep）|16 个只读工具 + `ToolContext.output\\\_dir` 路径拼接|**最大的实质收益点**|
|上下文压缩 + 工具输出落盘|无（靠 context\_payload 预注入 + max\_steps=8 硬截断）|新增，直接收益|
|`store=` 持久记忆|`rag/vector\\\_store.py` 自研 SQLite 双通道向量库|领域性强，**建议保留为工具**，别换成 langgraph store|
|`interrupt\\\_on` 人工审批|无；非法编辑靠 `\\\_apply\\\_edits` 容错 + `failed\\\_edits` 事后报告|**直接治 results.md §8h 的病**|
|LangSmith / 流式 trace|`llm/tracing.py:69` 自研 `raw\\\_json` + `view\\\_md` + README 索引|⚠️ 需并行保留，否则审计口径断裂|
|shell / skills / MCP|无|用不到，建议关掉|
|模型必须支持 tool calling|手写客户端，只读 content|**必须替换**|

**一个关键设计判断**：Deep Agents 让**模型自己决定**何时委派哪个 subagent（通过 `task` 工具）。
本项目要的是「每轮必须依次跑完 model→layer→optical→solution」。所以正确的用法是
**外层保留确定性流水线，每个角色各自 `create\\\_deep\\\_agent(...).invoke(...)`**，
而不是交给一个主 agent 自主编排——否则阶段保证会丢，`results.md` 的 per-run 口径全部失效。

\---

## 3\. 三条决定性事实（实测/已核实）

### 3.1 ✅ 模型侧具备 tool calling 能力（这条我实测过了）

对 `configs/llm.json` 里配置的 Paratera `DeepSeek-V4-Pro-0813` 端点发了一次真实的
OpenAI 风格 `tools=\\\[...]` + `tool\\\_choice=auto` 请求：

```
HTTP OK
has\\\_tool\\\_calls: True
tool\\\_calls: \\\[{"function": {"arguments": "{\\\\"value\\\\": 1}", "name": "ping"}, ...}]
finish\\\_reason: tool\\\_calls
```

也就是说，只要换成 `ChatOpenAI(base\\\_url="https://llmapi.paratera.com/v1", ...)`
（或 `langchain-deepseek`），模型侧没有任何障碍。**当前的手写客户端是唯一在丢弃
`tool\\\_calls` 的地方。**

### 3.2 ❌ Python 解释器是硬冲突

|事实|证据|
|-|-|
|`deepagents` 要求 `requires\\\_python <4.0,>=3.11`|PyPI 元数据（0.7.28，classifier 覆盖 3.11–3.14）|
|real 模式必须 `torch==1.12.1+cpu`|`scripts/env/requirements-real.txt`|
|torch 1.12 不支持 3.11+|`scripts/env/setup\\\_real.sh:7` 显式检查 `<= (3,10)`，并注释「torch 1.12.1 does not support 3.11+」|
|当前 real venv = Python 3.10.22|`.venv-real/pyvenv.cfg`，由 uv 托管在 `.tools/python/cpython-3.10.22-linux-x86\\\_64-gnu`|
|系统只有 Python 3.14.4|`/usr/bin/python3.14` 是唯一系统解释器；3.14 满足 ≥3.11|
|`.venv-real` 里**没有 pip 模块**|`.venv-real/bin/python -m pip` → `No module named pip`（uv 托管）|

**结论：`deepagents` 和 SimPhony real 模式无法共存于同一个解释器。**
要么拆双进程（SimPhony 变成进程外工具），要么不动依赖。

### 3.3 ⚠️ 依赖哲学冲突（这是个明确的取舍，不是技术障碍）

现在 `requirements.txt` 只有一行 `PyYAML>=5.1`，README 明确把
"no LangGraph, so the whole framework runs offline with a mock LLM" 当作卖点。
`deepagents` 的传递依赖是：`langchain>=1.4.6`、`langchain-core>=1.6.11`、
`langchain-anthropic`、`langchain-google-genai`、`langsmith`、`langgraph`、`pydantic v2`……
**从 1 个依赖变成几十个**，并且 `--provider mock` 的离线路径需要自己实现一个支持
`bind\\\_tools` 的 `BaseChatModel` 才能继续离线跑。

补充：uv 在 `.tools/uv`（0.13.1），但 uv 默认缓存 `\\\~/.cache/uv` 在当前沙箱下不可写，
任何安装都需要把 `UV\\\_CACHE\\\_DIR` 指到工作区内。

\---

## 4\. 需要改的东西：分层清单

### 4.1 环境层

* 新建 Python 3.12/3.13 venv（用 `.tools/uv`）承载 agent 框架；`.venv-real`（3.10）原样保留。
* 或者：**不引入依赖**，维持单解释器（见方案 B）。
* `requirements.txt`（纯 stdlib + PyYAML 的承诺）必须重写为两份：离线核心 + agent 框架增量。

### 4.2 LLM 客户端层（改动最彻底）

|文件|现状|需要做什么|
|-|-|-|
|`llm/clients.py:201`|`OpenAICompatibleClient`，urllib，只取 content|换成 `ChatOpenAI(base\\\_url=...)/langchain-deepseek`|
|`llm/clients.py:68/114`|`complete\\\_json\\\_messages` + `extract\\\_json\\\_object`（整响应即一个 JSON 对象）|该假设在 tool-calling 模式下不再成立，需改为 `with\\\_structured\\\_output()` / `response\\\_format`|
|`llm/clients.py:159`|`MockLLMClient` 返回固定 JSON|必须实现 `BaseChatModel` + `bind\\\_tools`，并能合成 `tool\\\_calls`，否则 120 个测试里的离线路径全挂|
|`llm/clients.py:80/90`|`\\\_trace\\\_begin` / `\\\_trace\\\_end` 钩子|LangChain 模型没有这两个钩子，要靠 callback handler 重建|
|`llm/config.py`|`LLMConfig`|映射到 `ChatOpenAI` 的 timeout / max\_retries / temperature / max\_tokens|

### 4.3 工具层（19 个）

* **16 个只读工具** → 用 `@tool` / `StructuredTool` 包装。好消息：`ToolSpec.input\\\_schema`
已经是标准 JSON Schema（`common/tool\\\_types.py:20`），schema 部分几乎零成本。
* **3 个 mutation 工具** → 建议挂 `interrupt\\\_on` 做人工审批；同时要把「原地改
`context.current\\\_hardware`、再读回」的语义（`agents/solution.py:189-211`）显式化，
LangChain 工具最好无状态，需要靠闭包/`InjectedState` 注入。
* **路径拼接** → `toolbox/common\\\_tool/\\\_report\\\_utils.py:\\\_load\\\_report` 依赖
`context.output\\\_dir`，应改为 deep agents 的 `FilesystemBackend(root=run\\\_dir)`，
让 agent 自己 `ls/read\\\_file/grep`，而不是把 evidence 预注入 prompt。
* **`photon\\\_tuner/rag/vector\\\_store.py`** → 保留为工具 `search\\\_history`，
双通道向量 + `hardware\\\_fingerprint` 是领域资产，不值得换掉。

### 4.4 Agent 层（4 + 1 个角色）

每个角色的固定结构（以 `agents/model\\\_level.py:56` 为例）：

```python
run\\\_react\\\_tool\\\_loop(..., max\\\_steps=8, allowed\\\_tool\\\_names=ALLOWED\\\_TOOLS)   # 证据阶段
complete\\\_json\\\_continuing\\\_messages(..., validate\\\_response=\\\_validate\\\_...)   # 严格 JSON 阶段
```

迁移后：

* 证据阶段 → deep agent 的单次 `invoke`。
* **严格 JSON 契约要重新建立**。现在 `validate\\\_response` 是硬校验（
`model\\\_level.py:105` 要求 `candidate\\\_layers` 是 list；`solution.py:253` 要求
`actions` / `link\\\_edits` 都是 list）。deep agent 的 subagent 返回的是**文本**，
所以要么用 structured output，要么让 subagent 把结果写成 artifact 文件由外层读取
—— 后一种和现有 `write\\\_json(output\\\_dir / "...\\\_result.json")` 完全一致，推荐。
* `max\\\_steps=8` → deep agents 有自己的循环终止策略，`max\\\_steps` 语义要重新定义。
* 重复调用检测（`react\\\_loop.py:115-122`，重复即终止）→ deep agents 没有等价物，
需要 middleware 或放弃。
* `EvidenceAnalystAgent`（`agents/evidence.py:46`）→ 第一个真正适合做 subagent 的角色。

### 4.5 编排层（`runtime/tuner.py`）

`tune()` 里这些是**研究可复现性的载体，建议留在 Python 宿主里，不要在迁移中交给模型**：

* `\\\_is\\\_better`（`:338`）keep-best 拒绝回归 —— 对应 `results.md` 里 vit\_base run\_0 的
+67% 回归被正确拒绝。
* `forbidden` fingerprint 集合 + `\\\_repair\\\_duplicate`（`:411`）兜底。
* `run\\\_dir` 命名（`\\\_next\\\_run\\\_dir:390`）、`trajectory.json`。
* 重要：`scripts/analysis/analyze\\\_sweep.py` 的 `LLM vs fallback` 标签**依赖
`\\\_repair\\\_duplicate` 的兜底行为**，不能静默去掉。

如果要把整条循环也交给 agent（planner + 每轮 4 个 subagent + 一个 `evaluate` 工具），
必须显式约束角色顺序（middleware 或 prompt），并接受轨迹形状变化。

### 4.6 产物与可观测性

* 现有产物契约：`iter\\\_NNN/\\\*.json` + `iter\\\_NNN/analysis/\\\*.json` + `trajectory.json`

  * `trace/{prompt\\\_calls.json,raw\\\_json/,view\\\_md/,system\\\_prompts.md,README.md}`。
* `llm/tracing.py:69` 的 `LLMTraceRecorder` 与 `tracing.py:28 DIALOGUE\\\_GROUPS`
是按 4 个角色的任务名硬编码的 —— 迁移后必须重建同等格式，否则 docs 里的审计证据链断裂。
* 建议：LangSmith 可以**额外**接，但不要替代本地 trace。

### 4.7 测试（120 个用例）

* **不能动的契约**：`tests/test\\\_pipeline.py:41/72` 断言 `summary\\\["iterations"]`、
`summary\\\["best"]\\\["objective"]`、以及 `run\\\_iteration\\\_001/iter\\\_000/workload.json`
的目录布局。
* **必须重写/删除**：`tests/test\\\_llm\\\_client.py`（6 个）、`tests/test\\\_llm\\\_tracing.py`（5 个）
直接测手写客户端与 recorder。
* **需要新造**：一个支持 tool calling 的假 chat model，撑起离线 CI。

### 4.8 脚本

* `scripts/experiments/sweep.py` 用 CLI 子进程驱动，只要 CLI 兼容就不用改；
但 `--provider` / `--llm-config` 的语义要扩展（langchain 模型串、base\_url）。
* `scripts/experiments/verify\\\_search\\\_space.py` + `evidence/` 与 agent 框架无关，零改动。

\---

## 5\. 三个方案

### 方案 A：完整引入 `deepagents` 包

* **前置**：新建 3.12 venv；SimPhony 必须**服务化**（进程外工具：CLI/JSON-RPC/MCP），
因为 3.10 的 torch 栈进不来。这是最大的架构改动——`adapter.evaluate()` 从进程内调用
变成进程外调用，评估延迟、错误处理、并发语义都要重做。
* **得到**：HITL、checkpoint/持久化、流式、上下文压缩、subagent 隔离、LangSmith 评测。
* **失去**：单解释器、纯 stdlib、mock 离线、历史可比性。
* 适合：目标是把它做成一个能对外服务的系统，而不只是一个研究 artifact。

### 方案 B：只移植 Deep Agents 的**架构模式**（不引入 langchain）— 推荐

在现有 runtime 上加四件事：

1. **规划**：一个 `write\\\_todos` 工具，写进 `run\\\_dir/todos.json`，让 agent 对一轮内的
证据收集显式规划。
2. **真正的文件系统工具**：把 `ls / read\\\_file / grep / glob` 挂在 `run\\\_dir` 上，
agent 自己去找 `energy\\\_detail.json` 里的 per-device 面积，而不是靠 prompt 预注入。
3. **上下文压缩**：超过阈值就把老 observation 落盘、只留摘要（现在只有 `max\\\_steps=8`
硬截断，浪费且不优雅）。
4. **subagent 隔离显式化**：4 个角色已经是独立消息栈，把它们改成「主 agent 通过
`task` 工具委派、但顺序由宿主保证」的形态。
* **保留**：Python 3.10、mock 离线、纯 stdlib、120 测试、产物布局、trace 格式、
确定性流水线、历史可比性。
* **收益**：直接针对 `docs/results/results.md` §4 记录的真实失效——
「没查面积就换 MZM 导致 area +2429%」「`in\\\_bit` 参数名幻觉」
「RAG 未被实际使用」——因为 agent 能自己去看文件。
* **代价**：HITL / LangSmith / checkpoint 要自己写（`interrupt\\\_on` 的等价物就是一个
approve 回调，几十行）。

### 方案 C：混合，最小改动

* 外层流水线完全不动，只把 `solution\\\_generation`（最需要探索、失败率最高的角色）
换成 deep agent，或者只做「文件系统工具 + 上下文压缩」。
* 适合先验证收益，再决定是否全量迁移。**风险最低，建议作为第一步。**

\---

## 6\. 不变量与风险

|风险|说明|
|-|-|
|**历史不可比**|换 loop 后 per-run 轨迹形状、`LLM vs fallback` 标签、proposal 形态都会变。`results.md` 需要在 docs 里标一个版本分界线。|
|**token 成本上涨**|当前基线：23 次调用 / 96k prompt / 42k completion 每轮（vit\_base）。Deep Agents 每步额外走 planning、可能读文件、subagent 结果回灌，预计显著上涨。**迁移前应先在单个 workload 上量化。**|
|**"离线可跑"卖点**|A 方案下变成"需要一个假 chat model"；B 方案下不变。|
|**torch 1.12.1 锁死 3.10**|`requirements-real.txt` 明确写了 "Do NOT re-resolve these"。想单进程单解释器 → A 直接不可行，除非放弃 real 模式或冒极大风险升级 torch 栈。|
|**回归风险集中区**|`tests/test\\\_pipeline.py` 的产物布局与 `summary` 契约、`analysis\\\_sweep.py` 的 fallback 标签、`tracing.py` 的审计格式——这三处是最容易在迁移中静默破坏的。|

\---

## 7\. 需要确认的问题

1. **你说的「deep agent」具体指哪个？**

   * (a) LangChain 的 `deepagents` 包（`create\\\_deep\\\_agent` / LangGraph 那套）→ 走方案 A/C；
   * (b) Deep Agents 那套**架构模式**（规划 + subagent + 虚拟文件系统 + 持久记忆）→ 走方案 B。
这两者要改的文件差别很大。
2. **迁移动机是哪个？** （决定值不值得付代价）

   * (a) 想要 HITL / 可观测 / 生产化生态 → A；
   * (b) 想治幻觉与非法编辑、让 agent 自己翻证据 → B/C，且性价比最高；
   * (c) 想让架构更"深"、更容易加新角色（对应 `docs/design/agent\\\_architecture.txt` 的
"更底层 agent 扩展，规则"）→ B。
3. **能不能接受 real 模式的 SimPhony 变成进程外工具？** 这是方案 A 的前提条件。

