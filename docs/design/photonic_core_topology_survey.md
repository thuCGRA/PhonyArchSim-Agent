# 光子计算核拓扑扩展调研报告

> 用途:为「光子计算核拓扑扩展」DSE 提供可直接写入设计草案的候选条目。
> 目标仿真器:SimPhony(拓扑 = 两层任意有向无环图 DAG 网表:core 层 + node 层)。
> 现有基线:DPTC core(MZM 交叉阵列 + MRR 重路由 + DODE node)+ DODE node(y_branch + phase_shifter + 2×2 coupler + 平衡 PD)。

## 0. 调研方法与 DOI 核实说明

- **web_search 不可用**(返回 HTTP 401 API 认证失败),本报告所有文献均改用 **Crossref API**(`api.crossref.org`)与 **OpenAlex API**(`api.openalex.org`)逐一核实。
- 下文每个 DOI 都经过 `api.crossref.org/works/<doi>` 真实查询确认可解析;标题、期刊、卷/页、年份、作者均取自 Crossref 返回的元数据,非记忆臆造。
- **数值纪律**:摘要/元数据中可读到的数字(如 40 aJ/MAC、2 POPS、11 TOPS、24-node、294×)才被引用并标注来源;论文正文中的器件级参数(如 Vπ、插入损耗、PD 响应度、环 Q 值、PCM Δn)在本会话无法从可访问的摘要层获得时,一律标 **not reported**,不臆造。
- SimPhony 现有器件库(16 类):`mzm, dac, adc, tia, waveguide, mrr_rerouter, laser_splitter, mmi, micro_comb, on_chip_laser, off_chip_laser, phase_shifter, photodetector, coupler, y_branch, crossing`。
- 网表记法沿用 `photon_tuner/simulators/simphony/configs/design/cores/dptc.yml` 的格式:
  - `instances: iN: [device_type, variant_index]`
  - `nets: nK: [src, dst]`(可带 `(s)`/`(e)` 表示 splitter 端口)
  - `scaling_rules: iN: <R,C,H,W,N 表达式>`;`R=replication,C=cores/tile,H=core height,W=core width,N=wavelengths`。

---

## 1. 现有 DPTC 基线的文献锚点(相干 MZM/光电倍增点积引擎)

### C0. Hamerly et al.,「Large-Scale Optical Neural Networks Based on Photoelectric Multiplication」

- **论文**:R. Hamerly, L. Bernstein, A. Sludds, M. Soljačić, D. Englund, *Physical Review X* **9**, 021032 (2019)。
- **DOI(已核实)**:[10.1103/PhysRevX.9.021032](https://doi.org/10.1103/PhysRevX.9.021032) ✅(Crossref: APS, vol 9, issue 2, 2019)
- **定位**:现有 DPTC 基线的**直接文献锚点**。「光电倍增(photoelectric multiplication)」= 用两个 MZM(或调制臂)分别编码输入 x 与权重 w,在光电探测器上做相干/差分解调,输出 ∝ w·x;再以交叉阵列/复用方式把多个点积汇总,构成可扩展的光学矩阵运算。摘要可核实结论:*「在保持光学器件能效优势的同时可扩展到大规模系统,网络性能有数量级提升」*(OpenAlex 摘要)。
- **关键器件参数**:本会话仅能核实摘要,**能效/吞吐量/精度具体数值 not reported**(需读全文;全文在 PRX 开放获取,建议后续按需提取)。
- **说明**:当前 `dptc.yml` 的 `ref: ""` 为空,建议把该字段回填为 `10.1103/PhysRevX.9.021032`,使基线有文献锚点。

---

## 2. 相干 MZM 点积的芯片级原始工作(补全锚点)

### C1. Xu et al.,「Optical coherent dot-product chip for sophisticated deep learning regression」(OCDC)

- **论文**:Shaofu Xu, Jing Wang, Haowen Shu, … Weiwen Zou, *Light: Science & Applications* **10**, 221 (2021)。
- **DOI(已核实)**:[10.1038/s41377-021-00666-8](https://doi.org/10.1038/s41377-021-00666-8) ✅(Crossref: LSA, vol 10, 2021)
- **定位**:硅基**相干点积芯片(OCDC)**,是相干 MZM 点积在 *Light: Science & Applications* 上的芯片级原始工作,与 Hamerly 互为「原理 vs 芯片」锚点。摘要要点(已核实):采用**光场承载运算、在完整实数域(含负值)运算**,而非仅正值域;通过**复用**让单片完成任意复杂度网络的矩阵乘与卷积;用**片上反传(in-situ backpropagation)标定**补偿工艺偏差;在 AUTOMAP 图像重建回归任务上与 32-bit 数字计算机可比。
- **node 级网表(平衡零差接收 node,「OHD」)**:

  ```
  instances:
    i0: [y_branch, 0]        # 信号与 LO 分离/合路
    i1: [coupler, 0]         # 2x2 耦合器(90° hybrid 简化)
    i2: [photodetector, 0]   # 平衡 PD 上臂
    i3: [photodetector, 0]   # 平衡 PD 下臂
  nets:
    n0: [i0, i1]             # signal 入 coupler
    n1: [i0, i1]             # LO 入 coupler
    n2: [i1, i2]
    n3: [i1, i3]
  insertion_loss_final_device: photodetector
  ```
  > 说明:OCDC 的核心是**相干(平衡)检测**做带符号乘法;若简化为单 PD 直接检测则退化为 DODE。

- **core 级网表(相干交叉阵列,光学行求和数据流)**:

  ```
  instances:
    i0: [on_chip_laser, 0]
    i1: [laser_splitter, 0]  # 扇出到 W 个输入列
    i2: [mzm, 0]             # 输入 MZM(编码 x),每列 1 个
    i3: [dac, 0]             # x 驱动
    i4: [mzm, 0]             # 权重 MZM(编码 w),每 (行,列) 1 个
    i5: [dac, 0]             # w 驱动
    i6: [node, 0]            # OHD 平衡零差 node,每行 1 个(行内光求和)
    i7: [tia, 0]
    i8: [adc, 0]
    i9: [waveguide, 0]
  scaling_rules:
    i0: 1                     # 或 R*C(每个 core 一个激光)
    i1: C*R                   # 扇出器
    i2: W*C*N                 # 每输入通道 1 MZM
    i3: W*C*N
    i4: H*W*R*C*N             # 权重 MZM 数 = 矩阵元素数
    i5: H*W*R*C*N
    i6: H*R*C                 # 每输出行 1 个平衡接收(行内相干求和)
    i7: H*R*C
    i8: H*R*C
    i9: H*W*R*C*N             # 互连
  nets:
    n0: [i0, i1(s)]
    n1: [i1(e), i2]           # 每列:splitter → 输入 MZM
    n2: [i3, i2]              # DAC → 输入 MZM
    n3: [i2, i4]              # 输入 MZM → 权重 MZM(串联)
    n4: [i5, i4]              # DAC → 权重 MZM
    n5: [i4, i6]              # 行内多路光求和后入平衡接收
    n6: [i6, i7]
    n7: [i7, i8]
  ```
  > 与现有 DPTC 的差异:DPTC 是 **node 每单元一个(缩放 H·W·R·C),逐单元探测后电学累加(temporal_accum_factor)**;OCDC/Hamerly 是**行内光求和,node 每行一个(缩放 H·R·C)**。这是可 DSE 的数据流轴:电学累加(ADC 多、对探测器噪声更鲁棒)vs 光行求和(ADC 少、相干噪声/相位误差更敏感)。

- **器件需求映射表**:

  | 器件 | 库内? | 说明 |
  |---|---|---|
  | mzm(输入/权重) | ✅ 已有 | 直接用 |
  | dac / adc / tia | ✅ 已有 | 直接用 |
  | on_chip_laser / laser_splitter | ✅ 已有 | 直接用 |
  | photodetector / coupler / y_branch | ✅ 已有 | OHD node 用 |
  | waveguide | ✅ 已有 | 互连 |
  | 90° optical hybrid | ❌ 需新增(可选) | 完整相干 IQ 解调需 90° hybrid;简化为 2×2 coupler 可先用现有库 |
  | **结论** | **零新增即可映射**(用 2×2 coupler 近似 hybrid) | |

- **权衡轴(相对 DPTC 基线)**:
  - 能效:**相近~更优**(光行求和减少 TIA/ADC 数量;但相干 LO 功耗计入激光)。
  - 面积:**更优**(node/ADC 从 H·W 降到 H)。
  - 速度:**相近**(受 MZM/PD/TIA 带宽限制,论文未见独立带宽数据 → not reported)。
  - 精度/噪声:**更差**(相干行求和对相位/路径误差敏感;OCDC 靠 in-situ 反传补偿偏差)。
  - 可扩展性:**相近**(规模受扇出与相干累积信噪比限制,摘要称「可复用单片做任意复杂度网络」)。

- **关键器件参数**:摘要未给器件级数值;**not reported**(MZM Vπ、PD 响应度、TIA 带宽、每 MAC 能耗等均需读全文)。

---

### C2. Sludds et al.,「Delocalized photonic deep learning on the internet's edge」(Netcast)

- **论文**:A. Sludds, S. Bandyopadhyay, Z. Chen, … D. Englund, *Science* **378**, 270–276 (2022)。
- **DOI(已核实)**:[10.1126/science.abq8271](https://doi.org/10.1126/science.abq8271) ✅(Crossref: Science, vol 378, page 270-276, 2022)
- **定位**:相干零差点积的**能效上限锚点**。乘法由「smart transceiver」的 MZM 权重 + 边缘端相干零差检测完成,即同一「光电倍增」原理的分布式版。摘要可核实数值:**40 aJ/MAC(<1 photon/MAC)**,98.8%/93% 识别精度;86 km 光纤场试验、3 THz 光带宽波分复用;毫瓦级边缘设备达到 TFLOPS 级运算。
- **node 级网表**:同 C1 的 OHD(相干零差平衡接收:coupler + 2× photodetector)。
- **core 级网表**:`on_chip_laser → mzm(权重, 远端 smart transceiver) → [waveguide/fiber] → node(OHD) → tia → adc`;WDM 由 micro_comb 或外置梳提供 N 个波长通道。
  ```
  instances: i0 on_chip_laser; i1 mzm(权重); i2 dac; i3 node(OHD); i4 tia; i5 adc; i6 waveguide
  scaling_rules: i1: H*W*R*C*N (权重); i3: H*R*C; i4/i5: H*R*C
  nets: n0 [i0,i1]; n1 [i2,i1]; n2 [i1,i3]; n3 [i3,i4]; n4 [i4,i5]
  ```
- **器件需求映射表**:全部在库内(mzm/dac/adc/tia/photodetector/coupler/on_chip_laser/waveguide);**零新增**。若做 WDM 需 micro_comb(库内已有)。
- **权衡轴(相对 DPTC)**:
  - 能效:**大幅更优**(40 aJ/MAC 是迄今最优点积能效之一)。
  - 面积:**更优**(边缘端只需探测+检测,权重在云端)。
  - 速度:**相近~更优**(TFLOPS 级)。
  - 精度/噪声:**更差**(<1 photon/MAC 意味着极低 SNR,依赖大冗余/纠错与统计平均)。
  - 可扩展性:**更优(分布式)**(把权重流式传输,规避片上大规模编程与读出的瓶颈)。
- **关键器件参数(已核实,来自摘要)**:40 aJ/MAC;<1 photon/MAC;98.8%/93% 精度;86 km;3 THz WDM。器件级 Vπ/响应度 **not reported**。

---

### C3. Youngblood,「Coherent Photonic Crossbar Arrays for Large-Scale Matrix-Matrix Multiplication」

- **论文**:Nathan Youngblood, *IEEE Journal of Selected Topics in Quantum Electronics* **29**(2), 1–11 (2023)。
- **DOI(已核实)**:[10.1109/JSTQE.2022.3171167](https://doi.org/10.1109/JSTQE.2022.3171167) ✅(Crossref: IEEE JSTQE, vol 29, page 1-11, 2023)
- **定位**:**相干交叉阵列 + 零差检测**做大规模矩阵-矩阵乘法,直接与 DPTC 同族。摘要要点(已核实):混合光电架构,用光子交叉阵列 + 零差检测;绕过「高速电子读出」与「权重频繁重编程」两大瓶颈,在大矩阵极限下显著降低能耗与延迟。权重以非易失方式存储(该课题组背景为相变材料 GSST;但本会话仅能核实摘要,**权单元具体器件 not reported**)。
- **node 级网表**:`i0 mzm(或 phase_shifter, 权单元)` + `i1 coupler` + `i2/i3 photodetector(平衡)`;即「权单元 + 平衡零差接收」。
- **core 级网表**(相干交叉阵列,行零差读出):
  ```
  instances: i0 on_chip_laser; i1 laser_splitter; i2 mzm(输入 x); i3 dac; i4 node(含权+平衡PD); i5 tia; i6 adc
  scaling_rules: i2: W*C*N; i4: H*W*R*C (每 (行,列) 一个权单元+探测); i5/i6: H*R*C
  nets: n0 [i0,i1]; n1 [i1,i2]; n2 [i3,i2]; n3 [i2,i4]; n4 [i4,i5]; n5 [i5,i6]
  ```
- **器件需求映射表**:
  | 器件 | 库内? | 说明 |
  |---|---|---|
  | mzm / phase_shifter(权单元) | ✅(若用热光/电光权) | 相变(GSST)权单元则需新增 |
  | photodetector/coupler | ✅ | 平衡接收 |
  | dac/adc/tia/laser/splitter/waveguide | ✅ | |
  | phase_change_cell(GSST/GeSbTe) | ❌ 需新增(若采用非易失权) | |
- **权衡轴(相对 DPTC)**:
  - 能效:**更优**(免频繁重编程 + 免高速电子读出)。
  - 面积:**相近**。
  - 速度:**更优**(矩阵-矩阵乘并行度)。
  - 精度/噪声:**相近~更差**(零差相干,受相位噪声影响)。
  - 可扩展性:**更优**(「大矩阵极限」下优势,摘要明确)。
- **关键器件参数**:摘要未给;**not reported**。

---

### C4. Zhu et al.,「Coherent General-Purpose Photonic Matrix Processor」

- **论文**:Zheyuan Zhu, Alireza Fardoost, Fatemeh Ghaedi Vanani, Andrew B. Klein, Guifang Li, Shuo S. Pang, *ACS Photonics* **11**, 1189–1196 (2024)。
- **DOI(已核实)**:[10.1021/acsphotonics.3c01694](https://doi.org/10.1021/acsphotonics.3c01694) ✅(Crossref: ACS Photonics, vol 11, page 1189-1196, 2024)
- **定位**:基于**相干模拟光子核**的通用矩阵处理器(MPU),做**带符号乘法**;可重构性与存储交给数字电子;配合误差管理策略,实现矩阵乘、矩阵求逆、有限差分偏微分方程求解。摘要要点(已核实)。
- **node 级网表**:相干核 = MZI(2× phase_shifter + 2× coupler)或 MZM+平衡接收,做带符号乘。
- **core 级网表**:`laser → splitter → MZI/MZM 相干核阵列(带符号乘) → 平衡 PD → TIA → ADC`;数字域提供权重重构与累积。
- **器件需求映射表**:phase_shifter/coupler/mzm/photodetector/dac/adc/tia/laser/splitter/waveguide 全在库内;**零新增**。
- **权衡轴(相对 DPTC)**:
  - 能效:**相近~更优**(相干模拟核)。
  - 面积:**相近**。
  - 速度:**相近**。
  - 精度/噪声:**更优(相对纯模拟)**——靠「误差管理策略」逼近浮点科学计算任务(摘要明确)。
  - 可扩展性:**相近**。
- **关键器件参数**:摘要未给;**not reported**。

---

## 3. 「Photonic tensor core」术语锚点 + 相变/非相干交叉阵列

### C5. Miscuglio & Sorger(术语锚点)+ Feldmann et al.(PCM 张量核芯片)

#### C5a. Miscuglio & Sorger,「Photonic tensor cores for machine learning」

- **论文**:Mario Miscuglio, Volker J. Sorger, *Applied Physics Reviews* **7** (2020)。
- **DOI(已核实)**:[10.1063/5.0001942](https://doi.org/10.1063/5.0001942) ✅(Crossref: APL Reviews, vol 7, 2020)
- **定位**:「**photonic tensor core**」术语的直接锚点。WDM 光子并行 + 相变材料多态非易失记忆。摘要可核实数值:**2 POPS(peta-operations/s)吞吐**、**4-bit 精度**、电气数据处理性能高 1 个数量级、纯光学数据处理高 2–3 个数量级(OPS/J)。
- **node 级网表**:`i0 phase_change_cell(多态非易失权)` + `i1 photodetector`(WDM 通道求和)。
- **core 级网表**:`micro_comb(N 波长 WDM) → WDM 分/合(awg 或 mrr_rerouter) → 相变权交叉阵列 → PD 阵列`。
- **器件需求映射表**:
  | 器件 | 库内? |
  |---|---|
  | micro_comb | ✅ |
  | mrr_rerouter(做 WDM 路由) | ✅ |
  | photodetector | ✅ |
  | waveguide | ✅ |
  | phase_change_cell(多态非易失) | ❌ 需新增 |
  | awg(WDM mux/demux) | ❌ 需新增(可用 mmi+micro_comb 部分替代) |
- **权衡轴(相对 DPTC)**:
  - 能效:**更优**(非易失权,近零静态功耗,摘要明确「near-zero static power」)。
  - 面积:**相近~更优**(WDM 复用一条波导)。
  - 速度:**更优**(2 POPS)。
  - 精度/噪声:**更差**(4-bit,多态模拟权噪声)。
  - 可扩展性:**相近**。
- **关键器件参数(已核实,来自摘要)**:2 POPS、4-bit、电气×1 数量级/光学×2–3 数量级性能提升。Vπ/PCM 折射率对比度等 **not reported**。

#### C5b. Feldmann et al.,「Parallel convolutional processing using an integrated photonic tensor core」

- **论文**:J. Feldmann, N. Youngblood, M. Karpov, … W. Pernice, *Nature* **589**, 52–58 (2021)。
- **DOI(已核实)**:[10.1038/s41586-020-03070-1](https://doi.org/10.1038/s41586-020-03070-1) ✅(Crossref: Nature, vol 589, page 52-58, 2021)
- **定位**:相变材料(PCM)交叉阵列 + 孤子微梳 WDM 的**集成光子张量核**芯片。摘要层未给数字(Crossref/OpenAlex 均无摘要),本会话可核实信息仅标题/元数据。正文常见报道(2×2、4×4 卷积核、PCM 权、微梳 WDM)属正文内容,**此处不列为已核实数值,建议以全文为准**。
- **node 级网表**:`i0 phase_change_cell(权)` + `i1 photodetector`(WDM 求和)。
- **core 级网表**:同 C5a(`micro_comb → WDM 分/合 → PCM 交叉阵列 → PD 阵列`)。
- **器件需求映射表**:同 C5a;`phase_change_cell` ❌ 需新增,`awg` ❌ 需新增。
- **权衡轴(相对 DPTC)**:能效**更优**(非易失+WDM);面积**更优**(WDM 复用);速度**更优**;精度/噪声**更差**(PCM 漂移/多态噪声);可扩展性**相近**。
- **关键器件参数**:**not reported**(本会话未核实到正文数值)。

---

### C6. Tait et al.,MRR 权重库 / Broadcast-and-Weight(非相干 WDM)

- **论文 A**:A. N. Tait, M. A. Nahmias, B. J. Shastri, P. R. Prucnal,「Broadcast and Weight: An Integrated Network For Scalable Photonic Spike Processing」, *Journal of Lightwave Technology* **32**, 4029–4041 (2014)。
  - **DOI(已核实)**:[10.1109/JLT.2014.2345652](https://doi.org/10.1109/JLT.2014.2345652) ✅(Crossref: JLT, vol 32, page 4029-4041, 2014)
- **论文 B**:A. N. Tait, T. Ferreira de Lima, E. Zhou, … P. R. Prucnal,「Neuromorphic photonic networks using silicon photonic weight banks」, *Scientific Reports* **7**, 7430 (2017)。
  - **DOI(已核实)**:[10.1038/s41598-017-07754-z](https://doi.org/10.1038/s41598-017-07754-z) ✅(Crossref: Sci Rep, vol 7, 2017)
  - 摘要可核实数值:**24-node 循环网络、294 倍加速**(相对常规基准);提出「modulator-class 神经元」功耗分析。
- **定位**:非相干 WDM 广播 + **MRR 权重库(add-drop 微环做正权重)** 做点积,与相干 MZM 基线正交,是「非相干点积变体」的首要候选。
- **node 级网表(MRR 权重 node)**:
  ```
  instances:
    i0: [mrr_weight, 0]   # 可调 add-drop 微环权(需新增器件类型)
    i1: [photodetector, 0] # drop 端口探测
  nets:
    n0: [i0, i1]           # bus → ring → drop PD
  ```
- **core 级网表(广播-权重数据流)**:
  ```
  instances: i0 on_chip_laser; i1 micro_comb(N WDM); i2 mzm(输入调制); i3 dac;
            i4 node(MRR 权重+PD); i5 tia; i6 adc; i7 waveguide; i8 mrr_rerouter(路由)
  scaling_rules:
    i2: W*C            # 每输入通道 1 调制器
    i4: H*W*R*C        # 每 (输出行,输入列) 一个 MRR 权
    i5/i6: H*R*C
    i7: H*W*R*C*N
  nets: n0 [i0,i1]; n1 [i1,i2]; n2 [i3,i2]; n3 [i2,i4]; n4 [i4,i5]; n5 [i5,i6]
  ```
- **器件需求映射表**:
  | 器件 | 库内? | 说明 |
  |---|---|---|
  | micro_comb / mzm / dac / adc / tia / photodetector / waveguide / mrr_rerouter / on_chip_laser | ✅ 全部已有 | |
  | **mrr_weight(可调 add-drop 加权微环)** | ❌ **需新增** | 现有库只有 `mrr_rerouter`(路由用),缺「权重环」器件;这是关键新增项 |
- **权衡轴(相对 DPTC)**:
  - 能效:**相近**(无相干 LO;但 WDM 与环热调谐功耗)。
  - 面积:**更优**(WDM 复用一条波导 + 环尺寸小)。
  - 速度:**更优**(WDM 天然高并行)。
  - 精度/噪声:**更差**(只做正值权、环热串扰、波长漂移敏感;相干基线可带符号)。
  - 可扩展性:**更优**(广播-权重天然可扩展,摘要强调 scalable)。
- **关键器件参数(已核实)**:24-node、294× 加速。环 Q/Vπ/热调谐功耗等 **not reported**。

---

## 4. MZI 网格(酉矩阵核)与复值扩展

### C7. 通用酉矩阵:MZI 三角/矩形网格

- **论文**(4 篇,均经 Crossref 核实):
  1. M. Reck, A. Zeilinger, H. J. Bernstein, N. Bertani,「Experimental realization of any discrete unitary operator」, *Physical Review Letters* **73**, 58–61 (1994)。DOI [10.1103/PhysRevLett.73.58](https://doi.org/10.1103/PhysRevLett.73.58) ✅
  2. W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, I. A. Walmsley,「Optimal design for universal multiport interferometers」, *Optica* **3**, 1460 (2016)。DOI [10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460) ✅
  3. Y. Shen, N. C. Harris, S. Skirlo, … M. Soljačić,「Deep learning with coherent nanophotonic circuits」, *Nature Photonics* **11**, 441–446 (2017)。DOI [10.1038/nphoton.2017.93](https://doi.org/10.1038/nphoton.2017.93) ✅
  4. N. C. Harris, J. Carolan, D. Bunandar, … D. Englund,「Linear programmable nanophotonic processors」, *Optica* **5**, 1623 (2018)。DOI [10.1364/OPTICA.5.001623](https://doi.org/10.1364/OPTICA.5.001623) ✅
- **定位**:表达**任意 N×N 酉矩阵**的 MZI 网格(Reck=三角,需要 N(N−1)/2 个 MZI;Clements=矩形,深度最短,同样 O(N²) MZI)。Shen 2017 是其在深度学习上的里程碑(可编程 MZI 网格;正文报道 56 个可编程 MZI——**属正文内容,本会话未独立核实,建议以全文为准**)。
- **node 级网表(单 MZI node)**:
  ```
  instances:
    i0: [phase_shifter, 0]  # 内臂相移 θ
    i1: [phase_shifter, 0]  # 外臂相移 φ
    i2: [coupler, 0]        # 2x2 耦合器(前)
    i3: [coupler, 0]        # 2x2 耦合器(后)
  nets:
    n0: [i2, i0]
    n1: [i0, i3]
    n2: [i2, i1]
    n3: [i1, i3]
  ```
- **core 级网表(MZI 网格)**:
  ```
  instances: i0 on_chip_laser; i1 node(MZI); i2 phase_shifter(对角相移); i3 photodetector(输出读出); i4 mzm(输入编码,可选); i5 waveguide
  scaling_rules:
    i1: (N*(N-1)/2)*R*C      # Reck 三角网格 MZI 数;或 Clements 矩形同阶
    i2: N*R*C
    i3: N*R*C
    i5: O(N^2)*R*C
  nets: n0 [i0,i4]; n1 [i4,i1]; ...(网格互连,无交叉则需 crossing); n2 [...,i3]
  ```
- **器件需求映射表**:phase_shifter / coupler / photodetector / mzm / on_chip_laser / waveguide / crossing **全在库内**;**零新增**。
- **权衡轴(相对 DPTC)**:
  - 能效:**相近**(无 DAC/ADC 于每单元,但网格光损耗随深度累加)。
  - 面积:**更差**(O(N²) MZI + 互连;DPTC 也是 O(H·W) 但单元更简单)。
  - 速度:**更优**(纯无源干涉,速率受 PD 限制)。
  - 精度/噪声:**相近~更差**(相位误差沿网格累积;酉约束限制表达域)。
  - 可扩展性:**更差**(深度/损耗/相位标定随 N 增长困难)。
- **关键器件参数**:**not reported**(本会话仅核实题录)。

### C8. 复值 MZI 神经网络

- **论文**:H. Zhang, M. Gu, X. D. Jiang, … ,「An optical neural chip for implementing complex-valued neural network」, *Nature Communications* **12** (2021)。
- **DOI(已核实)**:[10.1038/s41467-020-20719-7](https://doi.org/10.1038/s41467-020-20719-7) ✅(Crossref: Nat Commun, vol 12, 2021)
- **定位**:用相位+幅度编码复值,通过光干涉执行复值运算(比实值更快收敛、可构造非线性决策边界;摘要已核实)。
- **node/core 级网表**:同 C7 的 MZI 网格(MZI node + MZI 网格 core),加复值读出。
- **器件需求映射表**:与 C7 相同,**零新增**。
- **权衡轴(相对 DPTC)**:精度/表达力**更优(复值域)**;面积/可扩展性**更差**(同 MZI 网格);能效/速度**相近**。
- **关键器件参数**:**not reported**。

---

## 5. 补充候选(可映射,但优先级低于上述)

### C9. Chen et al.,「Deep learning with coherent VCSEL neural networks」

- **论文**:Z. Chen, A. Sludds, R. Davis, … D. Englund, *Nature Photonics* **17**, 723–730 (2023)。
- **DOI(已核实)**:[10.1038/s41566-023-01233-w](https://doi.org/10.1038/s41566-023-01233-w) ✅(Crossref: Nat Photon, vol 17, page 723-730, 2023)
- **定位**:VCSEL 阵列 + 相干(零差)检测的点积;是 C2 Netcast 的片上化扩展。摘要层无数字(**not reported**)。
- **node 级网表**:`i0 vcsel(需新增,或 on_chip_laser 替代)` + `i1 coupler` + `i2/i3 photodetector`。
- **core 级网表**:`vcsel 阵列(权重/输入) → 相干混合 → 平衡 PD → TIA → ADC`。
- **器件需求映射表**:除 **vcsel(垂直腔面发射激光器)** ❌ 需新增外,其余在库内。
- **权衡轴(相对 DPTC)**:能效**更优**(VCSEL 低功耗、面发射密度高);面积**更优**;速度**相近**;精度/噪声**相近**;可扩展性**相近~更优**(面阵列)。
- **关键器件参数**:**not reported**。

### C10. Xu et al.,「11 TOPS photonic convolutional accelerator for optical neural networks」

- **论文**:X. Xu, M. Tan, B. Corcoran, … D. J. Moss, *Nature* **589**, 44–51 (2021)。
- **DOI(已核实)**:[10.1038/s41586-020-03063-0](https://doi.org/10.1038/s41586-020-03063-0) ✅(Crossref: Nature, vol 589, page 44-51, 2021)
- **定位**:孤子微梳 WDM 驱动的光子卷积加速器(标题可核实数值 **11 TOPS**)。WDM 权重由梳齿谱整形实现;属 WDM 卷积核变体,可与 C5/C6 的 WDM 路径合并 DSE。
- **node 级网表**:`i0 photodetector`(波长求和)+ 可选谱整形权。
- **core 级网表**:`micro_comb(N) → 输入调制 → WDM 谱整形(权) → 色散/延迟 → PD 求和`。
- **器件需求映射表**:micro_comb/mzm/photodetector/waveguide/tia/adc **在库内**;**谱整形权单元 / awg** ❌ 需新增。
- **权衡轴(相对 DPTC)**:速度**更优(11 TOPS)**;能效**相近~更优**;面积**更优(WDM)**;精度/噪声**更差**(梳齿功率波动);可扩展性**相近**。
- **关键器件参数(已核实,来自标题)**:11 TOPS;其余 **not reported**。

---

## 6. 领域综述锚点(仅题录 + DOI,用于设计草案引用)

| 论文 | 期刊/年 | DOI(已核实) | 定位 |
|---|---|---|---|
| B. J. Shastri, A. N. Tait, T. Ferreira de Lima, W. H. P. Pernice, H. Bhaskaran, C. D. Wright, P. R. Prucnal,「Photonics for artificial intelligence and neuromorphic computing」 | *Nature Photonics* **15**, 102–114 (2021) | [10.1038/s41566-020-00754-y](https://doi.org/10.1038/s41566-020-00754-y) ✅ | **领域锚点综述**(主) |
| P. L. McMahon,「The physics of optical computing」 | *Nature Reviews Physics* **5**, 717–734 (2023) | [10.1038/s42254-023-00645-5](https://doi.org/10.1038/s42254-023-00645-5) ✅ | 物理约束与效率上限 |
| G. Wetzstein, A. Ozcan, S. Gigan, …「Inference in artificial intelligence with deep optics and photonics」 | *Nature* **588**, 39–47 (2020) | [10.1038/s41586-020-2973-6](https://doi.org/10.1038/s41586-020-2973-6) ✅ | 光电混合 AI 综述 |
| W. Bogaerts, D. Pérez, J. Capmany, …「Programmable photonic circuits」 | *Nature* **586**, 207–216 (2020) | [10.1038/s41586-020-2764-0](https://doi.org/10.1038/s41586-020-2764-0) ✅ | 可编程光子电路综述 |
| H. Zhou, J. Dong, J. Cheng, … X. Zhang,「Photonic matrix multiplication lights up photonic accelerator and beyond」 | *Light: Science & Applications* **11**, 30 (2022) | [10.1038/s41377-022-00717-8](https://doi.org/10.1038/s41377-022-00717-8) ✅ | 光子矩阵乘法方法综述(PLC / MZI / WDM 三法,摘要已核实) |
| T. W. Hughes, M. Minkov, Y. Shi, S. Fan,「Training of photonic neural networks through in situ backpropagation and gradient measurement」 | *Optica* **5**, 864 (2018) | [10.1364/OPTICA.5.000864](https://doi.org/10.1364/OPTICA.5.000864) ✅ | in-situ 训练方法(OCDC/相干核标定参考) |

---

## 7. 明确排除 / 不建议映射到 DAG 网表的项

- **Ashtiani et al.,「An on-chip photonic deep neural network for image classification」, *Nature* **606**, 501–506 (2022), DOI [10.1038/s41586-022-04714-0](https://doi.org/10.1038/s41586-022-04714-0) ✅(已核实)**:
  片上**衍射**深度网络,前向传播是自由空间/波导内连续衍射,非离散 instances+nets 的 DAG 结构,**不适合直接映射到 SimPhony 两层 DAG 网表**。列入但标注「不可映射」。

- **产业白皮书 / 无学术 DOI 的项**(按任务约束**不列**):Lightmatter Mars、Ayar Labs 等(无可解析学术 DOI)。

---

## 8. 汇总:候选优先级与「需新增器件」清单

### 8.1 候选优先级(相对 DPTC 基线的 DSE 扩展价值)

| 优先级 | 候选 | 一句话理由 | 新增器件 |
|---|---|---|---|
| P0 | C1 OCDC / C0 Hamerly(相干 MZM 点积) | 现有 DPTC 的直接锚点 + 芯片级原始工作 | 无(可选 90° hybrid) |
| P1 | C6 MRR 权重库(Broadcast-and-Weight) | 非相干 WDM 正交变体,补齐「非相干」轴 | **mrr_weight** |
| P1 | C5 PCM 张量核(Miscuglio/Feldmann) | 「tensor core」术语锚点 + 非易失权 + WDM | **phase_change_cell**,(可选)awg |
| P2 | C7 MZI 网格(Reck/Clements/Shen) | 通用酉矩阵核,零新增器件 | 无 |
| P2 | C3 Youngblood 相干交叉阵列 | 相干交叉阵列最直接的矩阵-矩阵乘工作 | 无(若相变权则同 C5) |
| P3 | C2 Netcast / C4 MPU / C8 复值 / C9 VCSEL / C10 微梳卷积 | 能效锚点 / 通用矩阵 / 复值 / VCSEL / WDM 卷积 | vcsel、awg(按需) |

### 8.2 需新增器件(汇总,供 device_library_extension_draft 合并)

1. **mrr_weight**——可调 add-drop 加权微环(区别于现有 `mrr_rerouter` 路由环);C6 必需。
2. **phase_change_cell**——相变非易失多态权单元(GSST/Ge₂Sb₂Te₅);C5a/C5b/C3(非易失权版)必需。
3. **awg**——WDM 复用/解复用器;C5/C10 建议(可用 mmi+micro_comb 部分替代)。
4. **vcsel**——垂直腔面发射激光器(或用 on_chip_laser 替代);C9 需要。
5. **hybrid90**——90° 光混合器(完整相干 IQ 解调);C1/C2/C3 可选(现有 2×2 coupler 可近似)。

### 8.3 数据可靠性备注

- 所有 DOI 经 Crossref 实时核实(见各条目「已核实」标)。
- 摘要可核实数值已逐条标注来源;C0/C5b/C7/C9 等论文正文中的器件级数值**未在本会话读取全文**,均标 not reported,写入设计草案前建议对 P0/P1 候选做一次全文数值抽取(尤其 C1 OCDC 与 C6 MRR 的每-MAC 能耗、Vπ、环参数)。

---

*生成时间:调研会话内;数据源:Crossref / OpenAlex(web_search 不可用)。*
