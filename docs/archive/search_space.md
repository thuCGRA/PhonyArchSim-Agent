# 搜索空间（Search Space）与覆盖现状

本文档记录 photonic DSE 的完整搜索空间定义、基线配置、以及 5×5 扫描（
`runs/sweep/`）实际触达的维度，并分析「为什么搜索到的维度很小」。

权威来源（运行时以 `DesignSpace` 为准）：

* [`photon\_tuner/core/design\_space.py`](photon_tuner/core/design_space.py) —
`LAYERED\_DESIGN\_SPACE`（分层 v2，实际被 `build\_design\_space` 加载）。
* [`photon\_tuner/core/search\_space.py`](photon_tuner/core/search_space.py) —
`DEFAULT\_ARCH\_SEARCH\_SPACE`（旧平面格式，迁移合并到分层默认）。
* [`configs/search\_space.json`](configs/search_space.json) —
分层的候选集快照（`{"layers": {system/optical\_core/topology/device/device\_param}}`），
由代码 `LAYERED\_DESIGN\_SPACE` 生成；`device.candidates` 为器件库快照（仅供参考）。
* [`configs/initial\_hardware.json`](configs/initial_hardware.json) —
分层基线硬件（`arch.system / arch.optical\_core / arch.topology / arch.memory` + `devices`）。
* 器件候选在运行时从 SimPhony 器件库
`simphony/configs/devices/\*\*/\*.yml` 实时发现，避免硬编码两份。

**逻辑层 → 物理路径**（`hardware` dict 已按层拆分，无嵌套）：

|逻辑层|物理路径|
|-|-|
|system|`arch.system.{tiles, cores\_per\_tile, dataflow, op\_type}` + `arch.memory.\*`|
|optical\_core|`arch.optical\_core.{height, width, num\_wavelength, work\_freq, precision}`|
|topology|`arch.topology.{sharing\_policy, temporal\_accum, laser\_distribution}`|
|device|`devices.core.\*` / `devices.node.\*`（选型）|

\---

## 1\. 分层结构

搜索空间按物理分解为 4 个可搜索层 + 1 个固定层：

|层|语义|是否可搜索|
|-|-|-|
|`system`|系统架构：tiles / cores / dataflow / precision / memory|✅|
|`optical\_core`|光学计算核心：crossbar 几何 / 波长 / 时钟|✅|
|`topology`|拓扑与电路：sharing / temporal / laser|✅|
|`device`|器件选择：每类型选一个 library variant|✅（仅选型）|
|`device\_param`|器件参数：library cfgs|❌ 固定，只选型不改 cfgs|

\---

## 2\. 架构参数与候选值

### 2.1 system（系统架构层）

|参数|候选值|默认|mode|
|-|-|-|-|
|`tiles`|\[1, 2, 4, 8]|4|both|
|`cores\_per\_tile`|\[1, 2, 4]|2|both|
|`dataflow`|output\_stationary / weight\_stationary / input\_stationary|output\_stationary|both|
|`precision\_in\_bit`|\[2, 4, 8]（约束 ≤ DAC 精度）|4|both|
|`precision\_w\_bit`|\[2, 4, 8]|4|both|
|`precision\_out\_bit`|\[2, 4, 8]（约束 ≤ ADC 精度）|4|both|
|`hbm\_bandwidth`|\[512, 1024, 2048]|1024|both|
|`glb1\_cache\_size`|\[64, 256, 485, 1024, 4096]|485|real|
|`glb2\_cache\_size`|\[64, 256, 485, 1024, 4096]|485|real|
|`rf\_cache\_size`|\[64, 256, 485, 1024, 4096]|485|real|

### 2.2 optical\_core（光学计算核心层）

|参数|候选值|默认|
|-|-|-|
|`core\_height`|\[8, 12, 16]|12|
|`core\_width`|\[8, 12, 16]|12|
|`num\_wavelength`|\[8, 12, 16]|12|
|`work\_freq`|\[5, 10, 15, 20, 25]（约束 ≤ MZM 符号率）|5 GHz|

### 2.3 topology（拓扑与电路层）

|参数|候选值|默认|
|-|-|-|
|`sharing\_policy`|no\_sharing / adc\_shared / adc\_tia\_shared|adc\_shared|
|`temporal\_accum`|off / default / deep|default|
|`laser\_distribution`|H*W / H*W*N / H*W*N*C|H\*W|

\---

## 3\. 器件层候选（`devices`，实时发现）

|器件类型|候选 variant（默认加粗）|
|-|-|
|mzm|ACADEMIA\_EO\_MZM\_1x1, ACADEMIA\_EO\_MZM\_2x2, **LT\_MZM**, SL\_MZM|
|dac|DAC\_1, **DAC\_2**|
|adc|**ADC\_SAR\_1**（唯一）|
|tia|**TIA\_1**（唯一）|
|coupler|ACADEMIA\_DC\_2x2, Customized\_DC\_1x1, Customized\_DC\_2x2, **LT\_DC\_2x2**|
|phase\_shifter|ACADEMIA\_TO\_PS, Customized\_PS, ITO\_PS, **MEMS\_PS**|
|photodetector|ACADEMIA\_Analog\_PD, ACADEMIA\_Digital\_PD, **LT\_PD**|
|on\_chip\_laser|**Customized\_Laser**, LT\_Laser|
|waveguide|**WG\_Single\_Strip**（唯一）|
|mrr\_rerouter|**LT\_MRR\_REROUTER**（唯一）|
|laser\_splitter|**Customized\_Laser\_Splitter**（唯一）|
|micro\_comb|**LT\_Micro\_Comb**（唯一）|
|y\_branch|Customized\_Y\_Branch, **LT\_Y\_Branch**|

> ⚠️ 器件库面积跨度极大：`LT\_MZM` 5200 µm² vs `ACADEMIA\_EO\_MZM\_1x1` 736000
> µm²（141×）。这是 vit\_base run\_1 回归 -853.76% 的直接原因（LLM 未查面积即换）。

\---

## 4\. 基线配置（SimPhony LT 设计，`default\_hardware()`）

```
arch.system:  tiles=4, cores\_per\_tile=2, dataflow=output\_stationary, op\_type=MVM
arch.optical\_core: type=Crossbar, height=12, width=12, num\_wavelength=12,
              work\_freq=5 GHz, precision in/w/out=4/4/4
arch.topology: sharing=adc\_shared, temporal=default, laser=H\*W
arch.memory:  HBM bw=1024, GLB2/GLB1/RF cache=485 (block\_size=8)
devices.core: mzm=LT\_MZM, dac=DAC\_2, adc=ADC\_SAR\_1, tia=TIA\_1,
              on\_chip\_laser=Customized\_Laser, ...
devices.node: photodetector=LT\_PD, coupler=LT\_DC\_2x2, phase\_shifter=MEMS\_PS, ...
```

\---

## 5\. 搜索空间量级

* 架构层组合 = system(364500) × optical\_core(135) × topology(27)
≈ **1.33×10⁹**
* 器件组合 = mzm(4) × dac(2) × coupler(4) × phase\_shifter(4) ×
photodetector(3) × on\_chip\_laser(2) ≈ **768**
* **总 ≈ \~1.0×10¹²**（约束 `work\_freq≤符号率`、`in\_bit≤DAC`、`out\_bit≤ADC`
会剪枝一部分不可行点）

\---

## 6\. 覆盖现状：已触达 vs 未触达维度

5×5 扫描（4 负载 × 5 次，`--iterations 1`）实际改动的维度：

### 已触达（5 个架构维 + 5 个器件维）

|维度|触达次数（近似）|杠杆强度|
|-|-|-|
|`work\_freq`（5→10）|\~11/20|强（+40\~61%）|
|`core\_height/core\_width`（12→16）|3/20|负载相关（vit\_small 有利 / vit\_base 面积灾难）|
|`dataflow`（output→weight）|3/20|弱（+3\~9%）|
|`mzm`（SL\_MZM / ACADEMIA\_EO\_MZM\_1x1）|\~10/20|弱（SL\_MZM +7.5%）|
|`coupler` / `dac` / `on\_chip\_laser` / `photodetector`|偶发|弱|

### 从未触达（12 个维度）

`tiles`、`cores\_per\_tile`、`num\_wavelength`、`precision\_in/w/out\_bit`、
`hbm\_bandwidth`、`glb1/glb2/rf\_cache\_size`、`sharing\_policy`、
`temporal\_accum`、`laser\_distribution`。

\---

## 7\. 为什么搜索到的维度很小（根因分析）

1. **`--iterations 1` = 每次运行只有一步贪心提案。** 5×5 扫描其实是 20 个
相互独立的「单步」搜索，不是 20 个收敛搜索；每次只从基线走一步，自然
只动一个瓶颈方向。
2. **诊断窄 → 杠杆窄。** `model\_agent` + `layer\_agent` 的诊断文本对所有负载
几乎都是「energy/switching dominates」，这只能映射到 work\_freq / 几何 /
器件三类杠杆；诊断里从不把 `tiles`、`precision`、`memory`、`sharing`
列为 root cause，于是这些维度从不被提出。
3. **光学 agent 只看 link budget → 只会器件选型。** 它不评估架构几何。
4. **无「多样化」压力。** solution agent 贪心优化单一诊断瓶颈，没有
ε-greedy / UCB / 随机探索等机制去覆盖 17 维架构空间。
5. **RAG 在 1 迭代模式下是空的。** `\_commit\_case` 在提案评估之后才写库，
迭代 1 检索时 `retrieved\_cases==\[]`，没有任何历史 case 引导探索新维度。
6. **`\_repair\_duplicate` 的回退顺序固定且偏 work\_freq。** 遇到重复提案时按
`work\_freq → num\_wavelength → core\_width → core\_height → tiles` 顺序 step，
使 work\_freq 被单调推高（5→10→15→…），进一步挤占其它维度。

> 结论：搜索策略「\*\*能找到最强单杠杆 work\_freq，但覆盖窄\*\*」。这是本次扫描
> 的诚实性边界；多迭代收敛实验 + RAG 检验将量化该边界能否被缓解。

\---

## 8\. RAG 机制（供检验）

* 存储：`HistoryVectorStore`，SQLite 库 `runs/<run>/history.db`。
* 提交：`tuner.\_commit\_case`（评估后）写一条 case（bottleneck 描述 + 硬件 +
解法 + before/after 指标 + bottleneck\_state）。
* 检索：`solution\_agent.propose()` 调
`store.search(state.retrieval\_description, hardware, top\_k)`，双通道向量
（75% bottleneck 语义 + 25% 硬件状态，`HashingEmbeddingModel`），余弦相似度
top-k，结果以 `retrieved\_cases` 注入 solution 的 LLM 上下文。
* **激活条件**：`\_commit\_case` 在提案评估后才执行 → 迭代 1 检索为空，迭代
≥2 才会检索到历史 case。因此 1 迭代扫描从未行使 RAG。

