# 光子计算核拓扑扩展 + 搜索空间与 agent 闭环完善 · 草案

> 状态:**草案待审阅** — 未实施任何代码/配置改动。4 路拓扑文献调研已完成,§3/§4 已填入;待你审阅后再决定是否实施。
> 关联:器件库扩展草案见 `device_library_extension_draft.md`(§3 候选、§7 拓扑机制分析);本草案承接其 §7,并补充"搜索空间 + agent 角色/职责/闭环"的完善方案。
> 铁律:扩展必须有真实文献依据(DOI 可解析),禁止空造;实施前必须经你审阅批准。

## 0. 变更记录

| 日期 | 变更 | 位置 |
|---|---|---|
| 2025-10-21 | 建立草案骨架:§2 现状盘点、§5 搜索空间完善、§6 agent 闭环完善(基于代码实读) | — |
| 2025-10-21 | 4 路拓扑文献调研返回,填入 §3(MZI 网格/广播-权重/微梳WDM·PCM/交叉阵列DPE)、§4(两层耦合矩阵),确定 §5.1 枚举值,补全 §7 待决问题 | §3/§4/§5.1/§7 |

## 1. 目标与约束

- **目标**:在"器件层扩展"之外,把**计算核拓扑结构(器件如何连接、什么计算范式)**也纳入真实、可搜索的扩展维度,并让两层协同(拓扑决定需要哪些器件,器件库供给拓扑所需器件);同时补齐搜索空间与 agent 角色/职责/闭环,使整个 DSE 闭环真实收敛、可复现。
- **拓扑扩展方式**:与 SimPhony 原生机制一致——新增 `configs/design/{nodes,cores,sub_architectures,architectures}/*.yml` 网表模板,并把"模板选择"变成搜索空间里的一个白名单枚举参数;`build_sub_arch`(`photon_tuner/simulators/analytic.py:187`)按参数加载模板,analytic 与 real 两条路径自动同步。
- **来源约束**:每个拓扑模板对应 ≥1 篇已发表真实文献,`ref` 填可解析 DOI;网表结构(instances+nets)必须能从论文描述的电路拓扑合理推导,参数不臆造。
- **有效性约束**:新模板相对现有 DPTC 交叉阵列基线,必须在 ≥1 个真实权衡轴(能效/面积/速度/精度/插损/可编程性/保持功耗)上更优、≥1 轴上更差,互不支配。
- **schema 约束**:网表必须是 DAG,`nets` 端点必须在 `instances` 中;使用到的器件类型必须落在能量公式覆盖内(`dac`/`mzm`/`phase_shifter`/`photodetector`/`on|off_chip_laser`/`adc`/`tia` + 通用 static/dynamic power),否则需先做器件层扩展;每模板需有 `insertion_loss_final_device`。
- **agent 约束**:拓扑模板是**白名单枚举**,LLM 只能从枚举中选择,不得自由改写网表(与 `photon_tuner/core/topology.py` 现有"白名单语义枚举"原则一致)。

## 2. 现状盘点(代码实读结论)

### 2.1 拓扑机制现状

- SimPhony 拓扑 = 层次化 4 级 + **任意 DAG 网表**(instances+nets+replication+scaling_rules),消费端是 networkx 拓扑排序求最长插损路径 + 按层级布局估面积(详见 `device_library_extension_draft.md` §7)。
- **当前只有一对固定模板**:DPTC core(相干 MZM 交叉阵列 + MRR 重路由 + DODE node)+ DODE node(y_branch+PS+2x2 coupler+平衡 PD)。
- 拓扑推断层是 `StaticTopologyProvider`(`photon_tuner/core/topology_spec.py`),只从 `hardware["devices"]` 派生一条**固定**光路(`OPTICAL_PATH_ORDER` 硬编码),`TopologyProvider` 协议预留了未来 `SimphonyTopologyProvider`。

### 2.2 搜索空间现状(`configs/search_space.json`,5 层)

| 层 | 参数 | 语义 |
|---|---|---|
| system | tiles / cores_per_tile / dataflow / precision_in·w·out_bit / hbm_bandwidth / glb1·glb2·rf_cache_size | 平铺、数据流、精度、存储 |
| optical_core | core_height / core_width / num_wavelength / work_freq | 交叉阵列几何与时钟 |
| topology | sharing_policy / temporal_accum / laser_distribution | **电路级参数**(ADC/TIA 共享、时间累加、激光分配),非图结构 |
| device | laser_source / splitter_kind + candidates 快照 | 实例级器件类型切换 + 变体快照(文档性) |
| device_param | (固定) | 库 cfgs 固定 |

**缺口**:
1. **图结构不是搜索维度**——拓扑模板固定 DODE+DPTC,无 `core_kind`/`node_kind` 参数;
2. 现有 "topology" 层命名与语义错位(实为"电路层"),新增"结构模板"维度时需澄清命名;
3. 器件变体 candidates 快照与运行时发现已通过 `EXCLUDED_SEARCH_VARIANTS` 过滤,但"拓扑所需器件类型是否齐备"无任何校验。

### 2.3 agent 系统现状(五角色确定性流水线)

`photon_tuner/runtime/orchestrator.py`:`model → architecture → topology → device → solution → (gate) → evaluate → 迭代`

| 角色 | 职责 | 输出 | 工具(只读为主) |
|---|---|---|---|
| model | 全局瓶颈层识别 | candidate_layers | summarize_* + compare/select_base |
| architecture | 层级根因(计算/存储/通信/缓冲/器件) | bottleneck_state | inspect_layer + summarize_* |
| topology | **损耗/功率预算诊断**(固定拓扑) | loss/power 诊断 | summarize_topology / inspect_light_path(只读) |
| device | 器件变体选择 | recommended_link_edits | list/inspect_device + summarize_device_* |
| solution | 提议下一次配置 | actions + link_edits | 只读 + 提议(modify/step_hardware_parameter、modify_device_parameter) |

- 编辑是**确定性应用**:`_apply_edits` 只认 `modify_hardware_parameter` / `step_hardware_parameter`(架构参数)与 `modify_device_parameter`(器件变体),LLM 不直接改硬件 dict。
- gate = 8 行协议(`photon_tuner/agents/evaluate.py`):deterministic 行 1(模式)/4(证据源)/5(编辑一致性)+ LLM 判定行 2/3/6/7 + 行 8 决策(accept / request-info / re-execute)。
- 循环:keep-best + forbidden-fingerprint 去重 + `_repair_duplicate` 兜底 + RAG 案例提交 + `trajectory.json`。

**闭环缺口**:
1. **gate 决策不生效**:`_run_gate` 只写 `gate_report.json`,行 8 的 request-info / re-execute 被丢弃,不触发重试或证据补齐;
2. **终止条件单一**:仅 `max_iterations` 固定上限,无"无改进早停 / 收敛阈值 / 预算上限";
3. **拓扑 agent 职责缺失**:只诊断固定拓扑的损耗/功率,不参与模板选择;无拓扑模板编辑工具;`StaticTopologyProvider` 是占位实现;
4. **结构合法性无 gate 行**:若引入模板切换,当前 gate 不校验"新模板能否被仿真器消费(器件类型覆盖、DAG 合法)";
5. **两层未耦合**:拓扑模板需要哪些器件类型,与器件库是否供给,无静态校验或证据探针。

## 3. 拓扑模板候选(真实文献调研)

> 每候选 = 论文+DOI、node/core 两级网表结构(instances+nets+scaling_rules)、器件需求映射、权衡轴、YAML 草图。

### 3.1 MZI 酉矩阵网格(Clements / Reck / Shen / Zhang)— ✅ 调研完成

**公共原子单元 = 可调 2×2 MZI**(所有网格的通用节点):两个 50:50 耦合器夹一个内相移 θ(定分束比),输入端一个外相移 φ。SimPhony DAG 记法:

```
node 级网表(可调 MZI = 实现 2×2 酉矩阵 T(θ,φ)):
instances: ps_phi=[phase_shifter,0]  c1=[coupler,0]  ps_theta=[phase_shifter,0]  c2=[coupler,0]
nets:      m_in→ps_phi→c1.in1 ; n_in→c1.in2 ; c1.out1→ps_theta→c2.in1 ; c1.out2→c2.in2 ; c2→m_out,n_out
```

**核心结论(诚实)**:MZI 网格**无需新增任何物理 device_type**——`phase_shifter`+`coupler`(或 2×2 `mmi`)即可构成;可选新增的是**复合宏/变体**:`tunable_mzi` 宏、`phase_shifter` 热光 variant、`balanced_pd`。

| 候选 | 论文 + DOI | 结构与规模 | 权衡 vs DPTC |
|---|---|---|---|
| **Clements 矩形网格(首选)** | Clements et al., Optica 3, 1460 (2016), 10.1364/OPTICA.3.001460, arXiv:1603.08788 | N(N−1)/2 个 MZI,深度 **N**(最小),矩形对称;0.2 dB/BS 损耗模型(引自 Carolan 2015, 10.1126/science.aab3642) | 万能可编程(显著优);面积 O(N²)(劣);大 N 插损陡(工程难点) |
| Reck 三角网格(对比基线) | Reck et al., PRL 73, 58 (1994), 10.1103/PhysRevLett.73.58 | N(N−1)/2 个 MZI,深度 **2N−3**;原始为体光学 | 同 Clements 但占位≈2×、插损路径不平衡;**不推荐作主拓扑** |
| **Shen SVD 接入(万能矩阵乘)** | Shen et al., Nat. Photonics 11, 441 (2017), 10.1038/nphoton.2017.93, arXiv:1610.02365 | 56 MZI + 213 PS;M=UΣV†(两个 SU(4) Clements + DMMC 对角衰减);热光相移 ~10 mW/调制器;DAC 240 通道 16-bit;元音 76.7% | 万能可编程 + 纯无源(计算零动态功耗);相位/热串扰限精度、热光重配置慢 |
| **Zhang 相干检测闭环(复值)** | Zhang et al., Nat. Commun. 12, 457 (2021), 10.1038/s41467-020-20719-7 | 8 模 56 PS;Reck 三角权重核 + 片上平衡相干检测(2×PD+TIA+ADC);50:50 MMI + TiN 热光;Iris 97.4% | 单激光+片上分光+相干检测闭环,复值维度翻倍;权重核仍三角(深度 2N−3) |
| **Butterfly/FFT 网格(快速变换)** | ~~Tang et al., "Ten-Port Unitary Optical Processor on a Silicon Photonic Chip", ACS Photonics 8, (2021), 10.1021/acsphotonics.1c00419~~;布局理论 Clements 2016(10.1364/OPTICA.3.001460)与 Miller 2013(10.1364/PRJ.1.000001) | **N·log₂N/2 个 MZI,深度 log₂N**(远浅于 Reck/Clements);但只实现**傅里叶/FFT 变换,非万能酉矩阵**;2025 年 Optica 10.1364/optica.559604(Ning 等)用结构化(butterfly)压缩做硬件高效张量核 | **器件数与深度最低(优)**;但**非万能**(劣)——只能做 FFT 域卷积/频谱处理,不能实现任意权重矩阵 |

> **订正(2025 复核)**:上表蝴蝶行的 Tang 2021(10.1021/acsphotonics.1c00419)经 Crossref 复核为
> 「Ten-Port Unitary Optical Processor」——是 10 端口**万能**酉矩阵处理器,**非**非万能 butterfly/FFT
> 网格,属错误引用,已在代码中撤回(`mesh_model.py`/`topology_space.py` 的 `ref` 置空并标注
> "citation under review")。MZI 数/深度公式本身来自经典 FFT 蝶形网络(Cooley-Tukey)。

**未报道(诚实标注)**:所有候选的 P_π、热响应时间、耦合器绝对插损均 **not reported**(仅 Clements 引用 0.2 dB/BS 模型;Shen 给 ~10 mW/调制器维持功率)。这些数值需在写库时留空或引用器件层候选补齐。

**关键工程点**:scaling 自变量是**空间模式数 M**(不是 `num_wavelength`)——需把 M 映射到 `core_width`/`core_height` 或新增 `num_modes` 参数。Butterfly 的 scaling_rules 用 `M·log₂M/2` 个 MZI、深度 `log₂M`。

### 3.2 广播-权重 + 微环权重库 — ✅ 调研完成

> 全量报告存于 `/home/fyq/epic/photonic_topology_survey_broadcast_weight.md`。3 篇核心论文全文经 arXiv HTML 逐字核对。

| 候选 | 论文 + DOI | 关键结构/参数 | 权衡轴 vs DPTC |
|---|---|---|---|
| **广播-权重循环核(架构奠基)** | Tait et al., J. Lightwave Technol. 32, 3427 (2014), 10.1109/JLT.2014.2345652 | 激光×N 广播→每 bank 微环权重→平衡 PD 差分求和→调制器神经元→WDM 回灌循环 | 非相干 WDM 正交变体;权重更新热调慢 |
| **硅光权重库(实验)** | Tait et al., Sci. Rep. 7, 7430 (2017), 10.1038/s41598-017-07754-z | 24 节点=576 权重(0.36 mm²)、180 fJ/SOP、4.4 mW/neuron、106 mW 总功率、294× 加速 | **面积 625 µm²/synapse(vs MZI 20,000)≈32× 更优;能效 180 fJ/SOP(vs 相干 830)更优** |
| **MRR 调制器神经元 node(全参数可得)** | Tait et al., Phys. Rev. Applied 11, 064043 (2019), 10.1103/PhysRevApplied.11.064043 | MRR 半径 11.5 µm、gap 200 nm、Q 14.5k→3.5k、热调 0.24 nm/mW、电调 ≤0.26 nm/mA、PD 0.76 A/W、片内增益 1.36、注入/耗尽带宽 6.25/40 GHz、sigmoid/ReLU/RBF 转移函数、片内 IL 18 dB | 完整神经元 node,可直接替换 DODE |
| 微环权重库分析 | Tait et al., IEEE JSTQE 22, (2016), 10.1109/JSTQE.2016.2573583 | 权重库 + 8 通道 | — |
| 4×4 MRR 交叉阵列 MAC | Ohno et al., ACS Photonics 9, 2614 (2022), 10.1021/acsphotonics.1c01777 | 4×4 交叉阵列 MAC | 单环参数 not reported |
| **对称微环交叉阵列(全参数可得,推荐)** | Tang/Ohno et al., Photonics Research 12, (2024), 10.1364/PRJ.520518 | MRR 半径 20 µm、FSR 4.4 nm、IL ~1.9 dB、热移相 19.3 mW/π、MZI ER 51 dB、芯片 3.7×2.4 mm²、Iris 93.3%、3.2×10⁵ ops/s@100 kHz;**支持片上反向传播训练** | WDM 使插损近似与 N 无关;权重更新慢(热调,100 kHz) |
| 9-bit 微环突触 | Zhang et al., Optica 9, 579 (2022), 10.1364/OPTICA.446100 | 抖动控制达 9 bit | 权重精度上探 |

**核心网表(可直接写 YAML)**:
- 广播-权重 core(`N=num_wavelength`):`laser×N`、`neuron×N`、`bank×N`(每 bank `mrr_modulator` 热调 ×N)、`wdm_mux×1`(N:1 合波,需新增)、`splitter 1×N`(y_branch 树 ×(N-1))、`bpd×N`(=2×photodetector+差分 tia);nets 呈"激光→WDM 合波→分光→各 bank 环 drop→平衡 PD→神经元→回灌"循环。
- 调制器神经元 node:2×photodetector + `mrr_modulator`(载流子注入变体,需新增)。
- 微环交叉阵列 core(对称版):`laser×N`、`mzi×2N`(编码前向 x / 反向 σ)、`mrr_modulator×N²`(add-drop 权重)、`photodetector×N`(列累加)、`wdm_mux×N`、`crossing×2N`;scaling_rules:`mrr_modulator=N²`、`mzi=2N`、`pd=N`。

**需新增器件类型(核心结论)**:① `mrr_modulator`(两变体:热调权重 / 载流子注入-耗尽调制;参数 radius/gap/Q/FSR/热调效率/电调效率)——**库内 `mrr_rerouter` 是开关/重路由,不能替代**;② `wdm_mux`/`wdm_demux`(AWG 或级联 MZI);③ 可选 `balanced_photodetector`、`saturable_absorber`/`gain_section`。复用:mzm/phase_shifter/photodetector/tia/y_branch/mmi/laser_splitter/crossing/coupler/dac/adc/laser/micro_comb。

**权衡轴 vs DPTC(全文级数字)**:面积(625 vs 20,000 µm²/synapse)、能效(180 vs 830 fJ/SOP)、插损扩展性(WDM 使插损近似与 N 无关)**更优**;权重更新速度(热调 µs–ms)、保持功耗(19.3 mW/π + 温漂标定)、编程复杂度**更差**。权重精度 4.1+sign → 5.1 bit@2Gbps → 9 bit(抖动)。

**not reported 清单**:JLT 2014 全部器件数值、Sci Rep 2017 的环 Q/FSR/单环调谐功耗/PD 灵敏度、JSTQE 2016/Ohno 2022 单环参数等——写入库前需回补原文(§1/§4 IEEE 闭源;Ohno 2022 有 arXiv 2106.04351)。

### 3.3 微梳 WDM 张量核 + PCM 非易失权重 — ✅ 调研完成

> 全量报告存于 `/home/fyq/epic/photonic_core_topology_survey.md`。**DOI 修正**:Xu 2021 = 10.1038/s41586-020-03063-0(非 -03063-4);Feldmann 2021 = 10.1038/s41586-020-03070-1(非 021-03230-9)。

| 候选 | 论文 + DOI | 关键结构/参数 | 需新增器件类型 |
|---|---|---|---|
| **微梳 WDM 卷积加速器** | Xu et al., Nature 589, 44 (2021), 10.1038/s41586-020-03063-0 | 11 TOPS、250k 像素卷积、88% MNIST;微梳时间/波长/空间三维交错;~1.58 pJ/MAC、8 bit、63 GHz | dispersive_delay、spectral_shaper、wdm_mux/demux |
| **PCM 光子张量核** | Feldmann et al., Nature 589, 52 (2021), 10.1038/s41586-020-03070-1 | ~10¹² MAC/s、>14 GHz、非谐振无源 PCM 透过率计算、孤子微梳;5 bit、17 fJ/MAC、59 TOPS/W、81 TMAC/s/mm² | **pcm_weight(GST)**、wdm_mux/demux |
| 暗脉冲微梳 9bit | Bai et al., Nat. Commun. 14, (2023), 10.1038/s41467-022-35506-9 | AlGaAsOI 暗脉冲微梳 FSR 91 GHz、9 bit 权重(record)、>1 TOPS/mm²、PAM-4@10Gbaud | mrr_weight、wdm_mux/demux |
| PCM 对称乘法单元 | Brückerhoff-Plückelmann et al., Nanophotonics 11, (2022), 10.1515/nanoph-2021-0752 | GST 衰减器 5 dB/100nm、布拉格 WDM MUX ER>61dB/IL<1dB、主+参考支实现有符号权重 | pcm_weight、wdm_mux/demux |
| PCM+WDM+RF 三维并行 | Dong et al., Nat. Photonics 17, (2023), 10.1038/s41566-023-01313-x | 三维并行度=100、心电猝死风险 CNN 93.5% | pcm_weight、wdm_mux/demux |
| 微梳+微环权重库 | Cheng et al., Nanophotonics 12, (2023), 10.1515/nanoph-2023-0298 | 51.2 TOPS、8 bit、六类情感 78.5% | mrr_weight、wdm_mux/demux |

**器件需求映射**:可复用 `micro_comb`/`mzm`/`on|off_chip_laser`/`photodetector`/`coupler`/`y_branch`/`waveguide`/`phase_shifter`(`mrr_rerouter` 可作 add-drop 复用);**需新增 4 类器件** = `pcm_weight`(GST 非易失权重单元)、`wdm_mux`/`wdm_demux`、`mrr_weight`(微环幅度权重调制器,或扩展 `mrr_rerouter`)、仅候选 1 另需 `dispersive_delay` + `spectral_shaper`。

**权衡轴 vs DPTC**:
- **WDM 微梳族**:吞吐/面积密度/波长并行**更优**;易失、热调慢(µs–ms)、需主动温漂锁定、插损高**更差**;精度 8–9 bit。
- **PCM 张量核族**:非易失零保持功耗、紧凑 in-memory、带宽 >14 GHz**更优**;精度仅 ~5 bit(GST 多电平+漂移)、重编程慢且耐久有限、吸收型固有插损**更差**。
- **DPTC(MZM 电光)仍优于两者**:重编程速度与权重精度(相干强度调制,无 PCM 漂移)。

**SimPhony 落地约定(scaling_rules)**:WDM 并行度用 `N=num_wavelength` 表达——`pcm_weight=N*M`、`spectral_shaper=N`、`mrr_weight=N`、`wdm_demux=1`、`micro_comb=1`;R/C/H/W 保持 tile/core/空间语义。node 层统一为"权重单元→coupler/y_branch 合束→photodetector"的 DAG 终点结构(与 DODE 同构,便于逐 node 替换权重单元做 DSE)。

**未报道(诚实标注)**:PCM 切换能耗/耐久循环数、微梳 FSR/线数(Xu/Feldmann 原文)均 not reported;性能数值部分取自开放综述(Zhou et al. Light Sci Appl 2022 Table 2)。

### 3.4 相干交叉阵列点积引擎基线锚点 + 其他可映射矩阵核 — ✅ 调研完成

> 全量报告存于 `docs/design/photonic_core_topology_survey.md`。与 §3.1/§3.3 重复的条目已省略。

**① 现有 DPTC 基线的文献锚点(回填 `ref`)**:Hamerly et al., "Large-Scale Optical Neural Networks Based on Photoelectric Multiplication", Phys. Rev. X 9, 021032 (2019), DOI 10.1103/PhysRevX.9.021032。光电倍增 = 双 MZM 编码 x/w + 相干/差分解调,输出 ∝ w·x。建议把 `dptc.yml` 的 `ref: ""` 回填此 DOI。

**② 相干 MZM 点积芯片(可零新增器件映射)**:

| 候选 | 论文 + DOI | 关键结构/参数 | 新增器件 |
|---|---|---|---|
| OCDC(相干点积芯片) | Xu et al., Light: Sci. Appl. 10, 221 (2021), 10.1038/s41377-021-00666-8 | 硅基相干点积,光场承载完整实数域(含负值),in-situ 反传标定,AUTOMAP 回归 32-bit 可比;**关键差异**:「行内光求和,node 每输出行 1 个(H·R·C)」vs 现有 DPTC「node 每单元(H·W·R·C)+ 电学/temporal 累加」——一个**数据流/累加轴**的 DSE 差异 | 无 |
| Netcast 相干零差 | Sludds et al., Science 378, 270 (2022), 10.1126/science.abq8271 | **40 aJ/MAC、<1 photon/MAC**、98.8%/93% 精度、86 km、3 THz WDM(能效锚点) | 无 |
| 相干交叉阵列+零差 | Youngblood, IEEE JSTQE 29(2), 2023, 10.1109/JSTQE.2022.3171167 | 相干矩阵-矩阵乘,免频繁重编程、免高速电子读出 | 无(相变权则需 phase_change_cell) |
| 相干通用矩阵处理器 | Zhu et al., ACS Photonics 11, 1189 (2024), 10.1021/acsphotonics.3c01694 | 带符号乘 + 误差管理,能做矩阵求逆/PDE | 无 |
| photonic tensor core 术语锚点(WDM+PCM) | Miscuglio & Sorger, Appl. Phys. Rev. 7, 031404 (2020), 10.1063/5.0001942 | 2 POPS、4-bit、光学 2-3 个数量级提升 | phase_change_cell、awg |
| 相干 VCSEL | Chen et al., Nat. Photonics 17, 723 (2023), 10.1038/s41566-023-01233-w | 相干 VCSEL | vcsel |

**③ 综述锚点(题录+DOI 均核实)**:Shastri Nat. Photon. 15, 102 (2021) 10.1038/s41566-020-00754-y(主锚点);McMahon Nat. Rev. Phys. 5, 717 (2023) 10.1038/s42254-023-00645-5;Wetzstein Nature 588, 39 (2020) 10.1038/s41586-020-2973-6;Bogaerts Nature 586, 207 (2020) 10.1038/s41586-020-2764-0;Zhou Light Sci. Appl. 11, 30 (2022) 10.1038/s41377-022-00717-8;Hughes Optica 5, 864 (2018) 10.1364/OPTICA.5.000864。

**④ 明确排除**:Ashtiani Nature 606, 501 (2022) 10.1038/s41586-022-04714-0(片上衍射,非 DAG,不适合映射);Lightmatter Mars / Ayar Labs 无学术 DOI(按约束不列)。

**⑤ 需新增器件类型(汇总)**:`mrr_weight`、`phase_change_cell`、`awg`、`vcsel`、`hybrid90`(见 §4 统一汇总)。

**数据可靠性**:OCDC / MRR 权重库 / VCSEL / PCM 的正文器件级数值(Vπ/IL/响应度/环 Q/PCM Δn)本会话未读全文,已标 not reported;写库前建议对 P0/P1 候选(OCDC、MRR 权重库)做全文数值抽取。

## 4. 拓扑 ↔ 器件 两层耦合矩阵

> 核心结论:**拓扑扩展反过来对器件层提出了第二轮扩展需求**——新拓扑需要 6 类器件库里没有的器件类型,必须与器件扩展(真实文献 + DOI)联合立项,两层不可割裂。

### 4.1 新增器件类型归一化(各调研代理命名不同,统一如下)

| 统一名 | 别名(各代理) | 功能 | 哪类拓扑必需 | 与库内现有器件关系 |
|---|---|---|---|---|
| `mrr_weight` | mrr_modulator | 微环**权重调制器**(热调权重 / 载流子注入-耗尽两个变体) | 广播-权重、微环交叉阵列、微梳 WDM 族 | **`mrr_rerouter` 是开关/重路由,不能替代** |
| `pcm_weight` | phase_change_cell | 相变(GST/GSST)非易失多态权重单元 | PCM 张量核族 | 库内 `mrr_rerouter` 的 MRR_PCM 候选是"开关"语义,非权重单元 |
| `wdm_mux`/`wdm_demux` | awg | WDM 复用/解复用(AWG 或级联 MZI) | WDM 微梳族、广播-权重、微环交叉阵列 | 库内 `mmi`/`laser_splitter` 可部分替代,但非波分 |
| `vcsel` | — | 面发射激光器 | Chen 2023 相干 VCSEL | 库内只有 on/off_chip_laser |
| `hybrid90` | — | 90° 光混合器(相干 IQ 解调) | 相干零差族(Netcast/Youngblood)可选 | 2×2 coupler 可近似,但非严格 90° |
| `dispersive_delay`+`spectral_shaper` | — | 色散延迟线 + 频谱整形 | 仅 Xu 2021 微梳卷积 | 无对应 |

**复合原语(可选,非新物理器件)**:`tunable_mzi`(=2×coupler+2×phase_shifter 宏)、`balanced_pd`(=2×photodetector+差分 TIA 宏)、`phase_shifter` 热光 variant——以"宏 + 变体索引"入库,而非新增物理器件类型。

### 4.2 拓扑模板 → 器件需求映射

| 拓扑模板(core_kind) | 复用库内器件 | 需新增器件类型 | 器件扩展是否已覆盖 |
|---|---|---|---|
| `crossbar_dptc`(现有基线) | 全部现有 | 无 | ✅ 已覆盖(回填 Hamerly 2019 `ref`) |
| `mzi_mesh`(Clements 矩形) | phase_shifter/coupler/mmi/mzm/laser/PD/TIA/ADC/DAC/waveguide/crossing | **无**(仅需热光 phase_shifter 变体 + tunable_mzi 宏) | ✅ 器件层基本够用 |
| `broadcast_weight`(广播-权重循环) | mzm/phase_shifter/PD/TIA/y_branch/mmi/laser_splitter/crossing/laser | **mrr_weight + wdm_mux/demux** | ❌ **需第二轮器件扩展** |
| `mrr_crossbar`(微环交叉阵列) | mzm/phase_shifter/PD/TIA/laser | **mrr_weight + wdm_mux/demux** | ❌ **需第二轮器件扩展** |
| `wdm_tensor_core`(微梳 WDM 卷积) | micro_comb/mzm/PD/coupler/waveguide | **pcm_weight(或 mrr_weight)+ wdm_mux/demux + dispersive_delay/spectral_shaper** | ❌ **需第二轮器件扩展** |
| `pcm_tensor_core`(PCM 张量核) | micro_comb/PD/coupler/waveguide | **pcm_weight + wdm_mux/demux** | ❌ **需第二轮器件扩展** |
| `coherent_homodyne`(相干零差/OCDC) | mzm/PD/TIA/ADC/DAC/coupler/laser | hybrid90(可选) | ✅ 器件层基本够用 |
| `vcsel_coherent`(相干 VCSEL) | PD/TIA/ADC/coupler | **vcsel** | ❌ 需第二轮器件扩展 |

### 4.3 两层联合立项结论

1. **器件扩展须分两轮**:第一轮(已完成草案 §3,8 类现有器件)支撑 `crossbar_dptc` 与 `mzi_mesh`;第二轮(新增 6 类器件)支撑 WDM/微环/PCM 族拓扑。
2. **第二轮器件扩展的调研对象**(真实文献 + DOI,复用第一轮方法):`mrr_weight`(锚 Tait 2019/PR2024)、`pcm_weight`(锚 Feldmann 2021/Brückerhoff 2022)、`wdm_mux/demux`(锚 Bai 2023/PR2024)、`vcsel`(锚 Chen 2023)、`hybrid90`、`dispersive_delay`+`spectral_shaper`。
3. **落地顺序**:先落 `mzi_mesh`(零新增,最快)+ 回填 DPTC 的 Hamerly `ref`;再启动第二轮器件扩展,解锁 WDM/微环/PCM 族。

## 5. 搜索空间完善方案

### 5.1 新增"拓扑模板"维度

- **命名澄清**:把现有 `topology` 层**重命名为 `circuit`(电路层)**,新增 `topology`(结构层)放模板选择,消除"拓扑层实为电路参数"的错位。
- **新增参数**(白名单枚举,调研已确定枚举值):
  ```json
  "topology": {
    "label": "topology (计算核结构模板层)",
    "params": {
      "core_kind": {
        "kind": "enum",
        "values": ["crossbar_dptc", "mzi_mesh", "broadcast_weight", "mrr_crossbar",
                   "wdm_tensor_core", "pcm_tensor_core", "coherent_homodyne", "vcsel_coherent"],
        "map": ["arch", "topology", "core_kind"], "mode": "both"
      },
      "node_kind": {
        "kind": "enum",
        "values": ["dode_mzi", "tunable_mzi", "mrr_neuron"],
        "map": ["arch", "topology", "node_kind"], "mode": "both"
      }
    }
  }
  ```
  > 枚举值对应 §3 候选:`crossbar_dptc`=现有基线、`mzi_mesh`=Clements/Reck/Shen、`broadcast_weight`=Tait 广播-权重、`mrr_crossbar`=Ohno/PR2024 微环交叉阵列、`wdm_tensor_core`=Xu 2021、`pcm_tensor_core`=Feldmann 2021、`coherent_homodyne`=OCDC/Netcast、`vcsel_coherent`=Chen 2023。首批落地建议只开**零新增器件**的 `mzi_mesh` + `coherent_homodyne`(其余等第二轮器件扩展,见 §4.3)。
- **接线**:`hardware_ops.py` 增加 `core_kind`/`node_kind` 到 `ARCH_PARAM_KEYS` 与 step/modify/normalize;`build_sub_arch` 按 `core_kind`/`node_kind` 选择加载 `configs/design/{cores,nodes}/<kind>.yml`;默认回落 `crossbar_dptc`/`dode_mzi`(现有行为不变)。
- **约束校验**(见 §6.4):切换后的模板必须满足 DAG、器件类型覆盖、`insertion_loss_final_device`。

### 5.2 器件层与拓扑层的同步

- device 层 candidates 快照按运行时发现刷新(已做过滤);新增"拓扑所需器件类型"的静态校验:加载模板时检查其 `devices` 表引用的类型都存在于器件库,缺失 → 报错并提示先做器件扩展。

### 5.3 证据探针

- `evidence/probe.py` 增加 `probe_topology`:枚举 `core_kind`/`node_kind` 模板,对每个模板返回结构诊断(DAG 合法性、器件类型覆盖、最长插损路径、器件计数表达式),供证据库与 gate 使用。

## 6. agent 角色 / 职责 / 闭环完善方案

### 6.1 角色定义(职责边界明确,均为白名单枚举,禁止自由改网表)

| 角色 | 职责(新增以 ⭐ 标出) | 输出(新增字段以 ⭐ 标出) | 工具 |
|---|---|---|---|
| model | 全局瓶颈层识别(不变) | candidate_layers | 不变 |
| architecture | 层级根因(不变) | bottleneck_state | 不变 |
| topology | ⭐ (a) 当根因指向"计算范式/并行度/数据流与拓扑不匹配"时,评估并建议切换 `core_kind`/`node_kind`;(b) 损耗/功率预算诊断(现有) | ⭐ `recommended_template`(core_kind/node_kind + rationale);loss/power 诊断 | ⭐ 加 `list_topology_templates` / `inspect_topology_template`(只读) |
| device | 在**选定模板约束下**选器件变体(现有 + 受模板器件需求约束) | recommended_link_edits | 不变 |
| solution | 提议下一次配置;⭐ 可提议 `modify_topology_template` | ⭐ actions 中允许 `modify_topology_template` | 不变(提议制) |
| gate/evaluate | ⭐ 结构合法性行(见 6.4) | 8 行 + 结构行 | 不变 |

### 6.2 新增编辑工具(确定性应用,与现有同机制)

- `modify_topology_template(core_kind, node_kind)`:白名单枚举切换,由 solution 提议、orchestrator 确定性应用(`_apply_edits` 增加该工具分支);非法值/缺失模板 → 计入 `failed_edits`。
- 只读检查工具 `list_topology_templates` / `inspect_topology_template` 给 topology agent。

### 6.3 让 gate 真正闭环

- **行 8 决策接入循环**:`accept` → 采纳 next_hardware;`request-info` / `re-execute` → 对 solution 阶段触发**一次补证/重提**(重跑 solution agent 或注入 gate 失败原因作为附加上下文),仍失败则记入 trajectory 并跳过该步(不崩溃)。当前 `_run_gate` 只写报告、丢弃决策,需改为把 decision 回传循环控制。
- **结构合法性 deterministic 行**:模板切换后,`_evaluate` 前硬校验(模板名合法、器件类型覆盖、DAG 合法、面积/IL 可算),失败 → 视为 `re-execute` 且该步不进历史 best。

### 6.4 终止/收敛条件(闭环的"出口")

- 在 `max_iterations` 之外增加:
  1. **无改进早停**:连续 `patience` 轮 best 的 objective 无改善(且已至少探索 `min_iterations` 轮)→ 停;
  2. **收敛阈值**:`|obj_prev − obj_best| / obj_best < ε` 连续若干轮 → 停;
  3. **可选预算上限**:评估次数/墙钟时间。
- 三者或的关系,默认 `max_iterations` 仍兜底。

### 6.5 证据与可复现闭环

- RAG 案例键(`hardware_fingerprint` / 检索描述)加入 `core_kind`/`node_kind`,使相似拓扑的历史案例可被检索;
- `trajectory.json` 记录每次 `modify_topology_template` 与 gate decision,保证轨迹可复现;
- `probe_topology` 结果写入证据,供 gate 行 4/6 引用。

## 7. 待决问题(写入/实施前需确认)

1. **拓扑模板枚举值(已确定,见 §5.1)**:8 个 `core_kind` + 3 个 `node_kind`。需确认:**首批只落地零新增器件的 `mzi_mesh` + `coherent_homodyne`(含回填 DPTC 的 Hamerly `ref`),还是等第二轮器件扩展一次到位?**
2. **"topology 层重命名为 circuit"是否采纳**:涉及 `search_space.json` 与 `hardware_ops.py`/`design_space.py` 的 map 路径同步;若不想大改,也可在现有 `topology` 层下直接加 `core_kind`/`node_kind` 两个 param(语义混在电路层内,但不改名)。
3. **第二轮器件扩展是否立项(两层耦合关键)**:新增 6 类器件 `mrr_weight`/`pcm_weight`/`wdm_mux·demux`/`vcsel`/`hybrid90`/`dispersive_delay+spectral_shaper`,复用第一轮"真实文献 + DOI + 草案"方法(见 §4.3)。这是 WDM/微环/PCM 族拓扑的前提。
4. **gate 接入循环的力度**(补证一次 vs 多轮重试)与早停参数(patience/ε)默认值;
5. **是否接受"LLM 只能白名单枚举选模板、不得自由改网表"硬边界**;
6. **MZI 网格的空间模式数 M 如何映射**:映射到 `core_width`/`core_height` 还是新增 `num_modes` 参数(§3.1 工程点)——影响 MZI 网格的 scaling_rules 写法。

## 8. 调研来源与可靠性说明

- 4 路拓扑调研由独立代理完成,全部 DOI 经 Crossref 逐条核实、doi.org 实测可解析;数值逐条对到论文摘要/全文(arXiv HTML/Europe PMC/OpenAlex),web_search 不可用时改用文献 API;
- 网表结构(instances+nets+scaling_rules)是"从论文电路图/描述到 SimPhony 模板"的**工程映射**,每候选已标注映射依据与不确定处;
- 未报道的参数一律标 "not reported";已标注的全文级参数(Shen 10 mW/相移器、Tait 625 µm²/180 fJ、PR2024 全参数等)可追溯原文;
- 三份全量调研报告:`/home/fyq/epic/photonic_core_topology_survey.md`、`/home/fyq/epic/photonic_topology_survey_broadcast_weight.md`、`docs/design/photonic_core_topology_survey.md`。
