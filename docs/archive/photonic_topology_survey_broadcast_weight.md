# 光子计算核拓扑调研报告：广播-权重(broadcast-and-weight)与微环权重库(MRR weight bank)

> 用途：可直接写入"光子计算核拓扑扩展"设计草案的候选条目。
> 调研方法说明：本会话 web_search 接口返回 401（认证失败），已改用 Crossref / OpenAlex / Semantic Scholar / arXiv 原生 API 与 arXiv HTML 全文。**所有 DOI 均已通过 `api.crossref.org` 逐条验证可解析**；所有数值均来自论文原文/arXiv 全文，未报道处明确标注 "not reported"，未臆造任何数值。

---

## 0. 候选条目总览

| # | 论文 | 类型 | DOI（已验证） | 本报告条目 |
|---|---|---|---|---|
| A | Broadcast and Weight: An Integrated Network For Scalable Photonic Spike Processing | 协议/架构奠基（仿真/理论） | 10.1109/JLT.2014.2345652 | §1 |
| B | Neuromorphic photonic networks using silicon photonic weight banks | 实验：4 节点循环广播-权重网络 | 10.1038/s41598-017-07754-z | §2 |
| C | Silicon Photonic Modulator Neuron | 实验：硅光调制器神经元（node 级） | 10.1103/PhysRevApplied.11.064043 | §3 |
| D | Microring Weight Banks | 权重库分析 + 8 通道实验拟合 | 10.1109/JSTQE.2016.2573583 | §4 |
| E | Si Microring Resonator Crossbar Array for On-Chip Inference and Training of the ONN | 实验：4×4 微环交叉阵列 MAC | 10.1021/acsphotonics.1c01777 | §5 |
| E' | Symmetric silicon microring resonator optical crossbar array（对称版） | 实验：4×4 对称微环交叉阵列 | 10.1364/PRJ.520518 | §5 |
| F | Silicon microring synapses enable photonic deep learning beyond 9-bit precision | 实验：9-bit 微环突触精度 | 10.1364/OPTICA.446100 | §6 |
| 辅1 | Feedback control for microring weight banks | 权重库反馈控制 | 10.1364/OE.26.026422 | §7 |
| 辅2 | Demonstration of scalable microring weight bank control… | 权重库可扩展标定 | 10.1063/1.5144121 | §7 |
| 辅3 | Photonic Multiply-Accumulate Operations for Neural Networks（综述，用于权衡轴数值） | 综述/建模 | 10.1109/JSTQE.2019.2941485 | §8 |

> 补充说明（供决策，非主条目）：Nature Commun. 2023 的 "In-memory photonic dot-product engine with electrically programmable weight banks"（10.1038/s41467-023-38473-x）虽然标题含 "weight bank"，但其权重器件是**相变材料 PCM** 而非微环，不属于本调研的 MRR 权重库范畴；如需对比可单列，本报告不纳入微环条目。

---

## 1. 候选 A：Broadcast and Weight（Tait 2014, JLT 32, 3427）

**DOI**：https://doi.org/10.1109/JLT.2014.2345652（Crossref 已验证，标题/期刊/年份一致）

- 作者：Tait, Nahmias, Shastri, Prucnal；J. Lightwave Technol. 32, 3427 (2014)。
- 地位：**"broadcast-and-weight" 协议的奠基论文**，首次系统定义该拓扑的协议、计算单元与波导媒介。本条目是"架构/协议"级候选，给出 node/core 网表时应以该协议描述为准，具体器件参数多为 not reported（该文为架构设计，非器件表征）。
- 核心思想（原文摘要 + 后续 Tait 系列文的一致性描述）：每个神经元输出分配一个唯一波长，经 WDM 复用后在共享波导总线上**广播**；接收端用可重构、连续取值的滤波（微环权重库 MRR weight bank）实现**加权**，再由总功率探测做**求和**，得到的电信号驱动神经元的电光非线性过程。支持波长复用（wavelength reuse）与循环网络。

### node 级网表（协议定义的"计算单元"，激光类神经元版本）
> 注：2014 原文针对的是**激光神经元（laser-class spiking neuron）**，这与本库基线 DPTC 的调制器/MZM 神经元不同；若要落地为库拓扑，建议以 §3 的调制器神经元替换激光神经元（见 §3 映射）。

```
instances:
  soma_ld    : on_chip_laser / off_chip_laser   // 每个神经元 = 可兴奋激光器（含增益，可能需饱和吸收体）
  weight[i]  : mrr_modulator (需新增, 热调变体)  // 每个输入波长一个微环权重, i = 1..N
  pd_sum     : photodetector                     // 总功率探测求和
nets:
  bus_in(多波长 WDM 广播) -> weight[i].through    // 各微环按波长抽取并加权
  weight[i].drop -> pd_sum.in                     // 加权分量汇入同一 PD 做非相干求和
  pd_sum.out(电) -> soma_ld.调制端                 // 电信号注入/调制激光神经元
  soma_ld.out(唯一波长 λn) -> 总线(回灌广播)
```

### core 级网表（broadcast-and-weight 全网络，N = num_wavelength）
```
instances:
  laser[i]    : off_chip_laser × N                // N 个波长载波 λ1..λN
  neuron[i]   : node(激光神经元或调制器神经元) × N
  bank[j]     : weight_bank × N                   // 每个接收神经元 j 一个权重库
                 (每 bank 内含 mrr_modulator(热调) × N)
  wdm_comb    : wdm_mux (N:1, 需新增) × 1          // 把 N 个神经元输出波长合入广播总线
  splitter    : 1×N 广播分束 (y_branch 树 × (N-1) 或 mmi)
  bpd[j]      : balanced photodetector × N         // = 2× photodetector + 差分 tia (可复用库内 tia)
nets:
  laser[i].λi -> wdm_comb.in[i]
  wdm_comb.out -> splitter.in
  splitter.out[j] -> bank[j].bus_in               // 广播到每个权重库
  bank[j].ring[i].drop -> bpd[j].in               // 权重 wij 累加到第 j 神经元
  bpd[j].out(电) -> neuron[j].in
  neuron[j].out(λj) -> wdm_comb.in[j]             // 形成循环(recurrent) + 可接输出
scaling_rules (每 core, N = num_wavelength):
  mrr_modulator(weight)    = N²
  mrr_modulator(neuron 输出)= N          // 若 neuron 为调制器神经元(§3)
  photodetector            = 2N         // 平衡探测
  laser                    = N
  wdm_mux                  = 1
  y_branch(splitter 树)    = N-1
  tia                      = N
  // 跨 tile 复制: 总量 × R × C; 布局: 权重库按 N×N 网格 (W=N 列, H=N 行)
```

### 器件需求映射（A）
| 功能 | 库内现有 | 需新增 | 备注 |
|---|---|---|---|
| 微环权重(热调滤波) | —（mrr_rerouter 是开关/重路由，非权重） | **mrr_modulator（权重变体，热调）** | 核心新增 |
| WDM 复用(N:1) | — | **wdm_mux / wdm_demux** | 核心新增 |
| 广播分束 | y_branch / mmi / laser_splitter | 无 | 树状级联 |
| 求和 PD | photodetector | 无（平衡探测需 2× PD + 差分 tia） | tia 已有 |
| 激光神经元(增益/饱和) | on_chip_laser / off_chip_laser | **saturable_absorber / gain_section（若做 spiking）** | 仅当追求激光类神经元 |

### 关键器件参数（A）
- 全部数值参数（微环 Q / FSR / 调谐功耗 / PD 灵敏度 / 插损 / 面积）：**not reported**（该文为架构级论文，摘要与后续论文均未给出 A 文自身的这些表征值）。
- 可确认的定性结论：支持 wavelength reuse 与大规模并行互连；计算单元 = 权重库 + 探测 + 神经元。

---

## 2. 候选 B：Neuromorphic photonic networks using silicon photonic weight banks（Tait 2017, Sci. Rep. 7, 7430）

**DOI**：https://doi.org/10.1038/s41598-017-07754-z（Crossref 已验证）；开放获取 arXiv:1611.02272；**全文已核对**。

- 作者：Tait, de Lima, Zhou, Wu, Nahmias, Shastri, Prucnal。
- 地位：**首个被报道的由微环权重库配置的循环( recurrent )硅光神经网络**。实验上观察了单节点 cusp 分岔（双稳）与双节点 Hopf 分岔（振荡），证明电路与连续时间循环神经网络 CTRNN 模型的同构。
- 实验规模：**4 节点循环网络，16 个可调微环权重**；权重矩阵通过微环滤波器的透过率实现；实验中神经元为片外光纤 MZM（调制器类），电学状态经低通滤波的跨阻放大器（TIA）表示。

### node 级网表（广播-权重"调制器神经元"节点，本文实验版）
```
instances:
  mrr_w[i] : mrr_modulator(热调, 需新增) × N        // 权重库：每输入波长一个微环
  pd_p     : photodetector                          // 正输入(兴奋)
  pd_n     : photodetector                          // 负输入(抑制)
  mzm_out  : mzm (库内已有)                         // 神经元输出调制器(实验中为片外光纤 MZM)
  tia_lp   : tia (库内已有, 兼低通)                 // 状态变量 s 的时间常数 τ 由电学低通决定
nets:
  bus_in(多波长) -> mrr_w[i].through/drop -> pd_p (加权正) / pd_n (加权负)
  pd_p/pd_n 差分光电流 -> tia_lp -> mzm_out.电调制端
  mzm_out.out(λj) -> 回灌总线(自反馈/互反馈)
```
> 注：实验用 off-chip 光纤 MZM 作神经元，库拓扑落地时可替换为 §3 的片上 MRR 调制器神经元。

### core 级网表（同 §1 广播-权重，本条目给出已实测的规模换算）
```
同 §1 core 网表结构；本条目实测参数支撑的 scaling_rules:
scaling_rules (每 core, N=24 例):
  mrr_modulator(weight) = N² = 576   // 原文: "24 MRR weight banks, 576 MRR weights"
  mzm / mrr_modulator(neuron) = N = 24
  laser                   = N = 24   // "24 laser wavelengths"
  photodetector           = 2N (平衡)
  wdm_mux                 = 1 (原文用 AWG 复用)
```

### 器件需求映射（B）
同 §1 映射表；特别地：**mzm 直接可用作神经元输出调制器**（库内已有），无需新增；权重仍依赖新增 mrr_modulator 与 wdm_mux。

### 关键器件参数（B，来自全文，已核对）
| 参数 | 值 | 来源 |
|---|---|---|
| 权重动态精度 | **4.1 + 1(符号) bit**（引用其前作 JSTQE 2016 权重库） | 正文 Metrics 段 |
| 每突触面积（微环权重） | **25 × 25 µm = 625 µm²/synapse** | 正文 |
| 24 节点权重库总面积 | 0.36 mm²（576 微环）；调制器 0.30 mm²（24 × 500 × 25 µm） | 正文 |
| 单神经元功耗 | **4.4 mW/neuron**（5% 墙插效率泵浦激光） | 正文 |
| 计算能效 | **180 fJ/SOP** | 正文 |
| 24 节点系统最低总功率 | 106 mW | 正文 |
| 信号带宽（限制） | 1 GHz（为抑制寄生振荡而限带） | 正文 |
| 反馈延迟 / 加速比 | t_fb = 47.8 ps；对 CPU 基准 **294×** 加速 | 正文 |
| 微环 Q / FSR / 每环调谐功耗 / PD 灵敏度 | **not reported**（本文为网络级演示，未单列这些器件参数） | — |
| 对照（相干 MZI 阵, Shen 2017） | 20 mW/neuron、830 fJ/SOP、200×100 µm=20,000 µm²/synapse | 正文（转引） |
| 对照（超导光电, Shainline） | 20 fJ/SOP、21 µm²/synapse、20 MHz | 正文（转引） |
| 对照（TrueNorth, Akopyan） | 240 fJ/SOP、6 µm²/synapse、1 kHz、65 mW/256M 突触 | 正文（转引） |

---

## 3. 候选 C：Silicon Photonic Modulator Neuron（Tait 2019, Phys. Rev. Applied 11, 064043）

**DOI**：https://doi.org/10.1103/PhysRevApplied.11.064043（Crossref 已验证）；开放获取 arXiv:1812.11898；**全文已核对**。

- 作者：Tait, Ferreira de Lima, Nahmias, Miller, Peng, Shastri, Prucnal。
- 地位：**片上硅光"调制器类神经元"的器件级论文**，是广播-权重架构缺失的最后一块拼图（此前权重网络已集成、神经元仍靠片外光纤 MZM）。演示了传递函数可配置、fan-in、抑制、时间分辨处理，以及关键的**自突触(autapse)级联能力**（通过观察双稳分岔证明微分增益 >1）。
- 器件构成：**两个平衡光电探测器（BPD）电连接到一个微环调制器（MRR modulator）**，即 O/E/O 通路。正 PD 光电流加入注入电流，负 PD 光电流从偏置端分流，二者与偏置电流 Ib 叠加后经自由载流子注入改变微环折射率，从而在泵浦波长 λn 上调制输出。

### node 级网表（可直接作为库里的"调制器神经元 node"，对应 DPTC 的 DODE node）
```
instances:
  pd_pos   : photodetector   // IN+ 端口(兴奋)
  pd_neg   : photodetector   // IN- 端口(抑制)
  mrr_out  : mrr_modulator (需新增, 载流子注入变体)  // 兼做非线性转移函数 + 波长转换
  heater   : (集成于 mrr_out 的 in-situ 热调, 用于粗波长锁定/偏置) 
nets:
  WDM 加权输入(多波长) -> pd_pos.in            // 兴奋 fan-in
  WDM 加权输入(多波长) -> pd_neg.in            // 抑制 fan-in
  pd_pos.光电流 + pd_neg.光电流(差分) + Ib -> mrr_out.调制端 (电)
  pump(λn, CW) -> mrr_out.in -> mrr_out.out   // 输出光, 同 λn, 可级联
```

### core 级网表
把 §1 的 `neuron[j]` 替换为本 node 即得广播-权重全片上网络；scaling_rules 与 §1 相同（mrr_modulator(neuron 输出) = N）。

### 器件需求映射（C）
| 功能 | 库内现有 | 需新增 | 备注 |
|---|---|---|---|
| 输出调制器(非线性+波长转换) | mzm（可替代，但本文为微环） | **mrr_modulator（载流子变体）** | 核心新增 |
| 平衡探测 | photodetector ×2 + 差分 tia | 无 | tia 已有 |
| 权重微环 / WDM | — | mrr_modulator(热调变体) + wdm_mux | 同前 |

### 关键器件参数（C，来自全文，已核对）
| 参数 | 值 |
|---|---|
| 微环调制器半径 / 耦合间隙 | **11.5 µm / 200 nm** |
| 波导 | 220 nm 厚 Si，宽 500 nm（直段加宽至 3 µm） |
| 泵浦/输入波长 | λn = 1544.8 nm；λ1 = 1546.4、λ2 = 1548.2 nm |
| 热调效率 | **0.24 nm/mW** |
| 电调(注入)效率 | **≤ 0.26 nm/mA**（阈值以上） |
| 品质因数 Q | **14.5k**（偏置下可降至 **3.5k**） |
| PD 响应度 | **0.76 A/W** |
| 光-光增益 | 片内 **1.36**；含耦合 2.16×10⁻²（受 18 dB 光纤-芯片插损主导） |
| 调制带宽 | 注入调制可达 **6.25 GHz**；耗尽调制可达 **40 GHz** |
| 转移函数种类 | sigmoid / ReLU / RBF / quadratic |
| 脉宽压缩 | 500 ps→324 ps（36%）；burst 25 ns→20.4 ns（19%） |
| 光纤-芯片插损 | 片内插损 18 dB；光栅耦合器 9 dB |
| FSR / 每环绝对调谐功耗 / PD 灵敏度(NEP) | **not reported** |

---

## 4. 候选 D：Microring Weight Banks（Tait 2016, IEEE JSTQE 22）

**DOI**：https://doi.org/10.1109/JSTQE.2016.2573583（Crossref 已验证）。

- 作者：Tait, Wu, de Lima, Zhou, Shastri, Nahmias, Prucnal。
- 地位：**微环权重库的规范化分析论文**。给出"独立加权(independent weighting)"的定量描述，建立**通道数 vs 功率代价(power penalty)**的权衡；揭示总线波导上的**相干多谐振器相互作用**对性能的显著影响；实验拟合一个 **8 通道权重库**。
- 本文是 §2/§3 中"权重精度 4.1 + 1(符号) bit"与"标定+前馈连续权重控制"的原始出处（被 Sci. Rep. 2017 引用为其 ref [35]）。

### node/core 网表
权重库本身就是广播-权重 core 中的 `bank[j]` 子结构：一条 through 总线串联 N 个微环（每个微环 add-drop，负责一个波长通道），drop 端口合并送入平衡探测。无独立 node 概念，可直接作为 §1 core 网表的子块。

### 器件需求映射（D）
同前：权重微环 = **mrr_modulator(热调变体)**（需新增）；合并/探测复用库内器件。

### 关键器件参数（D）
| 参数 | 值 |
|---|---|
| 实验权重库通道数 | **8 通道**（实验测量拟合） |
| 权重精度 | **4.1 + 1(符号) bit**（动态权重精度，被 Sci. Rep. 2017 转引） |
| 关键限制机理 | 总线波导上的**相干多谐振器相互作用**导致通道数↔功率代价权衡 |
| 微环 Q / FSR / 单环调谐功耗 / PD 灵敏度 / 面积 | **not reported**（需查原文 IEEE 全文，本会话仅能核对摘要与二次引用） |

---

## 5. 候选 E / E'：微环交叉阵列（MRR crossbar）矩阵核

### E：Ohno 2022, ACS Photonics 9, 2614
**DOI**：https://doi.org/10.1021/acsphotonics.1c01777（Crossref 已验证）；开放获取 arXiv:2106.04351。
- 地位：**首个全集成 4×4 微环交叉阵列原型做矩阵-向量乘（MVM）并演示图像分类**；提出用交叉阵列的**转置矩阵运算做片上反向传播（on-chip backpropagation）**，使同一芯片支持推理+训练。
- 结构：N×N 交叉阵列用 **N² 个微环表示矩阵**，**2N 个 MZI 生成前向/反向两个输入向量**；每个微环 add-drop 的 drop 端口透过率 = 矩阵元，列输出波导由 PD 做累加。前向/反向用不同输入/输出端口实现 W 与 Wᵀ，无需重配微环。
- 注意：该版为**非对称结构**，各路插损不平衡（前向路径 crossing 数 0..2N−2、反向 0..2N），是后续对称版要解决的主要问题。

### E'：Tang/Ohno 2024, Photonics Research（对称版）
**DOI**：https://doi.org/10.1364/PRJ.520518（Crossref 已验证）；开放获取 arXiv:2401.16072；**全文已核对**。
- 地位：**对称化微环交叉阵列**，消除各光路插损不平衡；演示 4×4 电路做 Iris 分类，并用模拟片上反向传播训练；另仿真 9×9 阵列做 MNIST 卷积。
- 对称设计：每条光路前向/反向都固定经过 **2N 个 waveguide crossing**，各路长度与插损一致。

### node 级网表（交叉阵列的"权重单元"，等价 node）
```
instances:
  mrr_w : mrr_modulator (需新增, add-drop 热调)   // 一个矩阵元
  cross : crossing (库内已有) ×2                  // 对称化插损(输出口补偿)
nets:
  行波导(λi, 编码 xi) -> mrr_w.in
  mrr_w.drop -> 列波导 -> pd_j (列累加)
  mrr_w.through -> 行波导(继续传播)
```

### core 级网表（N×N 交叉阵列，N = num_wavelength）
```
instances:
  laser[i]  : off_chip_laser × N                  // λ1..λN
  mzi[i]    : mzm / phase_shifter(库内已有) × 2N  // 前向 x 与反向 σ 的输入向量编码
  mrr[i][j] : mrr_modulator(add-drop) × N²        // 权重矩阵 W
  pd[j]     : photodetector × N                   // 列累加(乘加)
  wdm_mux[i]: wdm_mux (需新增) × N                // 把 N 波长合入第 i 输入口
  crossing  : crossing × (每路径 2N)              // 对称化
  te_filter : (偏振滤波, 可用 crossing/coupler 实现) × 2N
nets:
  laser 合波 -> wdm_mux[i].in
  wdm_mux[i].out -> mzi[i](编码 xi) -> 行波导 i
  行波导 i -> mrr[i][j].in ; mrr[i][j].drop -> 列波导 j
  列波导 j -> pd[j].in -> 输出 yj (累加结果)
  // 反向: 反向输入 σ 从另一组端口注入, 复用同一 mrr 阵列实现 Wᵀ·σ
scaling_rules (每 core, N=num_wavelength):
  mrr_modulator = N²
  mzi           = 2N
  photodetector = N
  laser         = N
  wdm_mux       = N
  crossing      = 2N·(路径数相关, 对称版为每路径 2N)
```

### 器件需求映射（E/E'）
| 功能 | 库内现有 | 需新增 | 备注 |
|---|---|---|---|
| 权重微环(add-drop) | — | **mrr_modulator（热调）** | 核心新增 |
| 输入向量编码 | mzm / phase_shifter | 无 | |
| WDM 合波 | — | **wdm_mux** | 核心新增 |
| 列累加 | photodetector | 无 | |
| 交叉/偏振滤波 | crossing / coupler | 无 | |

### 关键器件参数（E'，来自全文，已核对；E 若无特殊注明同架构）
| 参数 | 值 |
|---|---|
| 阵列规模 / 芯片面积 | 4×4；**3.7 × 2.4 mm²** |
| 微环半径 / FSR | **20 µm / 4.4 nm** |
| 微环插入损耗（谐振处） | **~1.9 dB** |
| 微环 Q（9×9 仿真假设） | **~3×10⁵** |
| 波导 | 440 nm × 220 nm，TE 传输损耗 **1.3 dB/cm** |
| 热移相器功耗 | **~19.3 mW/π** |
| MZI 消光比 | 最高 51 dB（最差 37.6 dB） |
| 微环 drop 消光比 | ~30 dB |
| 波长 | λ1–λ4 = 1549.00/1549.75/1550.50/1551.25 nm（间隔 0.75 nm） |
| 光纤-芯片耦合损耗 | ~2.9 dB/facet |
| 计算速度 | **~3.2×10⁵ ops/s @ 100 kHz 时钟**（受热调移相器限速） |
| 任务精度 | Iris 推理 **93.3%**；训练后 **91.1%**；MNIST 仿真 **93.4%** |
| PD 灵敏度 / 每环绝对调谐功耗 | **not reported** |

---

## 6. 候选 F：Silicon microring synapses beyond 9-bit precision（Zhang 2022, Optica 9, 579）

**DOI**：https://doi.org/10.1364/OPTICA.446100（Crossref 已验证）。

- 作者：Zhang, Huang, Peng, Bilodeau, Jha, Blow, de Lima, Shastri, Prucnal。
- 地位：**微环突触权重精度的记录**：用 **抖动(dithering)控制方案** 在光突触上实现 **9-bit** 精度（针对光神经网络普遍受串扰与谐振器高灵敏度限制导致低精度的问题），并数值仿真了提高突触精度对无线信号分类的收益。

### node/core 网表
与 §1/§2 权重库一致（微环权重 = mrr_modulator 热调变体 + 平衡探测 + WDM），本条目核心增量在**控制/标定**而非拓扑：需为每个微环权重引入 dithering 控制信号（可映射为 dac 驱动的控制回路）。

### 器件需求映射（F）
权重微环（mrr_modulator）+ wdm_mux（需新增）；控制回路复用 dac/adc/tia。

### 关键器件参数（F）
| 参数 | 值 |
|---|---|
| 权重精度 | **9 bit**（dithering 控制，记录值） |
| 微环 Q / FSR / 调谐功耗 / PD 灵敏度 / 面积 | **not reported**（需查原文） |

---

## 7. 辅助条目：权重库控制（写入草案"编程复杂度"轴依据）

- **Tait 2018, Opt. Express 26, 26422**：DOI 10.1364/OE.26.026422。基于光致电阻变化估计滤波器透过率的反馈控制，实现权重库多通道连续控制：**2 Gbps 信号上 5.1 bit 精度**；不做直接透过率测量仍达 **4.2 bit**。
- **Huang 2020, APL Photonics 5, 040801**：DOI 10.1063/1.5144121。标定过程**无需直接访问权重库输出、无需专门标定 I/O**，实现记录级精度/精确度与可忽略的通道间串扰；为大规模扩展提供可控面积与 I/O 成本。
- 结论：微环权重库的编程复杂度（温漂、标定、串扰）是真实且非平凡的问题，草案应把"每环 dithering/反馈标定回路"列为必要支撑（dac+adc+tia 复用，另需片上监测 PD 或光致电阻监测，**需新增 monitor 单元或复用 photodetector**）。

---

## 8. 权衡轴汇总（相对 DPTC 基线 = MZM 交叉阵列）

DPTC 基线为 MZM 交叉阵列（空间 fan-out/fan-in 非相干乘加），本报告 MRR 权重库/广播-权重与之逐轴对比：

| 权衡轴 | 微环权重库/广播-权重 vs DPTC MZM 交叉阵列 | 依据 |
|---|---|---|
| 权重更新速度 | **更差**：热调微环为 µs–ms 量级（PR2024 时钟 100 kHz；热移相器限速） | E' 全文 |
| 保持功耗 | **更差/需注意**：热调需持续功耗（~19.3 mW/π），且温漂需反馈回路；MZM 热移相器类似但微环对温度更敏感 | E'、辅1/辅2 |
| 面积 | **显著更优**：微环权重 625 µm²/synapse vs MZI 相位器 20,000 µm²/synapse（约 32×） | B 全文 |
| 能效 | **更优**：180 fJ/SOP vs 830 fJ/SOP（相干 MZI 网） | B 全文 |
| 插损/扩展性 | **更优**：WDM 广播使插损近似与 N 无关；空间交叉阵列插损随 N（分束树）增长 | B、E'（每 MRR 仅 ~1.9 dB） |
| 编程复杂度 | **更差**：需逐环标定 + 反馈（温漂/串扰）；但可达 4–5 bit，dithering 后 9 bit | 辅1、辅2、F |
| 计算吞吐 | **更优**：WDM 多波长并行（每波长 GHz 级、2 Gbps 演示），N² 权重共享 O(N) 波导 | B、辅1 |
| 综述级结论 | 光处理器在 >100 µm 尺寸、N>500 向量、≤4 bit 精度区间占优；能量>10²、速度>10³、密度>10² | 辅3（JSTQE 2019） |

---

## 9. 器件需求映射总表（写入草案的"需新增器件类型"清单）

| 库内现有 | 能否覆盖 | 结论 |
|---|---|---|
| mrr_rerouter | ❌ 仅开关/重路由 | **不能当权重/调制器用** |
| mzm | ✅ 神经元输出调制/输入向量编码 | 可复用（§2 神经元、§5 输入 MZI） |
| phase_shifter | ✅ MZI 相位、热调 | 可复用 |
| photodetector / tia | ✅ 探测/差分求和 | 平衡探测 = 2×PD + 差分 tia |
| y_branch / mmi / laser_splitter | ✅ 广播分束 | 树状 1:N |
| crossing / coupler | ✅ 交叉阵列布线 | 复用 |
| dac / adc | ✅ 权重控制/量化接口 | 复用 |
| laser（on/off_chip）、micro_comb | ✅ 波长载波源（micro_comb 可作多波长源） | 复用 |
| **需新增 #1** | — | **mrr_modulator**（两变体：`variant=热调权重` 与 `variant=载流子注入/耗尽调制`；参数含 radius/gap/Q/FSR/热调效率/电调效率） |
| **需新增 #2** | — | **wdm_mux / wdm_demux**（AWG 或级联 MZI；N:1 合波、1:N 分波） |
| **需新增 #3（可选）** | — | **balanced_photodetector**（若不想用 2×PD+tia 组合建模）；**saturable_absorber/gain_section**（仅激光类神经元） |

---

## 10. 建议：草案最优先落地的两条拓扑

1. **广播-权重循环核（§1 结构 + §3 调制器神经元 node + §2 实测规模换算）**
   - 最贴合"扩展现有 DPTC 基线"：把 DODE node 换/并为"MRR 调制器神经元 node（§3）"，core 层引入 wdm_mux + 1:N 广播 + N² 权重微环。
   - 优点：面积/能效/插损扩展性全面优于 DPTC；缺点：需新增 mrr_modulator + wdm_mux，且需权重标定/反馈回路（§7）。
2. **微环交叉阵列 MAC 核（§5，推荐对称版 E'）**
   - 前向推理 + 片上反向传播（转置复用）一套阵列，天然支持训练；器件少、规则网格、易配 scaling_rules。
   - 优点：结构规整、支持训练、Iris 93.3% 已验证；缺点：热调限速（100 kHz 级），高吞吐需换 EO/MOSCAP 移相器（文中自述）。

---

## 11. 未报道参数清单（诚实标注，避免草案臆造）

- §1（JLT 2014）全部器件数值：not reported。
- §2（Sci. Rep. 2017）微环 Q/FSR/单环调谐功耗/PD 灵敏度：not reported。
- §3（PR Applied 2019）FSR/单环绝对调谐功耗/PD 灵敏度(NEP)：not reported。
- §4（JSTQE 2016）Q/FSR/单环调谐功耗/PD 灵敏度/面积：not reported（仅核对摘要 + 二次引用）。
- §5-E（Ohno 2022）单环 Q/FSR/调谐功耗等：not reported（仅核对摘要 + E' 对架构的描述）。
- §6（Zhang 2022）除 9-bit 精度外其余器件参数：not reported。

> 若草案必须引用这些参数，建议回补原文 PDF（§1/§4 为 IEEE 闭源，§5-E 为 ACS 闭源但 arXiv 开放）后再填入，本报告不代填。

---

*本报告所有 DOI 均通过 Crossref 验证可解析；数值来源为 arXiv 全文（1611.02272、1812.11898、2401.16072）或 Crossref/OpenAlex/Semantic Scholar 元数据，未检索到来源的数值一律标注 not reported。*
