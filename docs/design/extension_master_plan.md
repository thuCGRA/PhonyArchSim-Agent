# 光子 DSE 扩展总清单:拓扑 + 器件 + 搜索空间 + agent

> 状态:**待审阅,未实施**。本文件是两份草案(`device_library_extension_draft.md`、`topology_and_agent_extension_draft.md`)的**一页总览**;明细证据见对应草案。铁律:每条扩展必须有可解析 DOI/arXiv 的真实文献依据,未报道参数标 "not reported",禁止臆造。

## 0. 一图总览

```
拓扑扩展(core_kind/node_kind) ──依赖──▶ 器件扩展(第一轮 8 类 + 第二轮 6 类新器件)
        │                                        │
        └──────────▶ 搜索空间新增拓扑模板维度 ◀───┘
                              │
                              ▼
                  agent 修改:topology 角色扩为"结构选择+预算诊断",
                  gate 接入循环,终止/收敛条件,证据闭环
```

## 1. 拓扑扩展清单(来源 + 器件需求 + 权衡 + 搜索空间贡献)

| core_kind | 拓扑 | 来源(DOI) | 需新增器件 | 与现有 DPTC 的权衡(优/劣) | 对搜索空间的贡献 |
|---|---|---|---|---|---|
| `crossbar_dptc` | 现有基线(相干 MZM 交叉阵列) | Hamerly 2019, Phys. Rev. X 9, 021032, 10.1103/PhysRevX.9.021032(**回填 `ref`**) | 无 | 基准(重编程快、权重精度高) | 基线,0 贡献 |
| `mzi_mesh` | Clements 矩形酉网格 | Clements 2016, Optica 3, 1460, 10.1364/OPTICA.3.001460 | 无(热光 PS 变体 + tunable_mzi 宏) | 万能可编程**优**;面积 O(N²)、热光慢**劣** | 新增「万能可编程性」轴 |
| `mzi_mesh_reck` | Reck 三角酉网格 | Reck 1994, PRL 73, 58, 10.1103/PhysRevLett.73.58 | 无 | 深度 2N−3、插损不平衡(劣) | 理论基线锚点 |
| `butterfly_fft` | Butterfly/FFT 快速变换 | ~~Tang 2021, ACS Photonics 8, 10.1021/acsphotonics.1c00419~~;理论 Clements 2016 / Miller 2013(10.1364/PRJ.1.000001);2025 结构化压缩 10.1364/optica.559604 | 无 | **N·log₂N/2 MZI、深度 log₂N(最低)优**;只实现傅里叶、**非万能**劣 | 新增「专用快速变换 vs 万能」轴(FFT 域卷积) |

> **订正(2025 复核)**:上表蝴蝶行的 Tang 2021(10.1021/acsphotonics.1c00419)经 Crossref
> 复核为「Ten-Port Unitary Optical Processor」——是 10 端口**万能酉矩阵处理器**,**非**非万能
> butterfly/FFT 网格,属错误引用。MZI 数/深度公式来自经典 FFT 蝶形网络(Cooley-Tukey);
> 光子 mesh 实现出处当前**无可靠 DOI**,代码中 `ref` 已置空并标注 "citation under review"。
| `broadcast_weight` | 广播-权重循环核 | Tait 2014, JLT 32, 3427, 10.1109/JLT.2014.2345652;Tait 2017, Sci. Rep. 7, 7430, 10.1038/s41598-017-07754-z | **mrr_weight** + wdm_mux/demux | 面积 625 µm²/synapse(32× 优)、180 fJ/SOP 优;权重热调慢劣 | 新增「面积密度 + 能效」轴 |
| `mrr_crossbar` | 对称微环交叉阵列 | Tang/Ohno 2024, Photonics Research 12, 10.1364/PRJ.520518;Ohno 2022, 10.1021/acsphotonics.1c01777 | **mrr_weight** + wdm_mux/demux | IL~1.9 dB、**片上反向传播训练**优;时钟 100 kHz 劣 | 新增「片上训练」轴 |
| `wdm_tensor_core` | 微梳 WDM 卷积 | Xu 2021, Nature 589, 44, 10.1038/s41586-020-03063-0 | pcm/mrr_weight + wdm + dispersive_delay/spectral_shaper | 11 TOPS、8 bit、吞吐/密度优;易失、插损高劣 | 新增「波长并行 + 吞吐」轴 |
| `pcm_tensor_core` | PCM 非易失张量核 | Feldmann 2021, Nature 589, 52, 10.1038/s41586-020-03070-1 | **pcm_weight** + wdm_mux/demux | 非易失零保持、59 TOPS/W 优;精度 ~5 bit、耐久有限劣 | 新增「非易失 in-memory」轴 |
| `coherent_homodyne` | 相干零差/OCDC 点积 | Sludds 2022, Science 378, 270, 10.1126/science.abq8271;Xu OCDC 2021, Light Sci. Appl. 10, 221, 10.1038/s41377-021-00666-8 | 无(可选 hybrid90) | **40 aJ/MAC、<1 photon/MAC**优;行求和数据流(ADC 减为 H·R·C) | 新增「能效锚点 + 数据流」轴 |
| `vcsel_coherent` | 相干 VCSEL | Chen 2023, Nat. Photonics 17, 723, 10.1038/s41566-023-01233-w | **vcsel** | 面发射、紧凑光源优;需新激光器件 | 新增「光源形态」轴 |

| node_kind | 节点 | 来源(DOI) | 需新增器件 | 说明 |
|---|---|---|---|---|
| `dode_mzi` | 现有点积节点(y_branch+PS+coupler+平衡PD) | Hamerly 2019(同上) | 无 | 基线 |
| `tunable_mzi` | 可调 2×2 MZI 节点 | Clements 2016 / Shen 2017(10.1038/nphoton.2017.93) | 无 | MZI 网格的原子单元 |
| `mrr_neuron` | MRR 调制器神经元节点(含片上非线性) | Tait 2019, Phys. Rev. Applied 11, 064043, 10.1103/PhysRevApplied.11.064043 | **mrr_weight**(载流子注入变体) | 全参数可得:Q 14.5k、0.76 A/W、sigmoid/ReLU |

## 2. 器件扩展清单

### 2.1 第一轮(8 类现有器件,支撑 `crossbar_dptc`/`mzi_mesh`/`coherent_homodyne`)

| 器件类型 | 候选 | 来源(DOI) | 关键参数 | 权衡轴 | 状态 |
|---|---|---|---|---|---|
| adc | ADC_SAR_90GS | ISSCC 2014, 10.1109/ISSCC.2014.6757477 | 90 GS/s、8b、667 mW、0.45 mm² | 速率 9× 解锁 work_freq 15-25;功耗 ×45 | ✅(ENOB 5.2-5.7 标注) |
| adc | ADC_SAR_4P8GS_LP | JSSC 2026, 10.1109/jssc.2026.3677867 | 4.8 GS/s、10.4 mW、28nm | 功耗 −30%;速率收窄 | ⚠️ **缺 area** |
| dac | DAC_3 | SSCL 2022, 10.1109/lssc.2022.3205884 | FoM 23.2、50 GS/s、9b、243 mW | FoM 优、速率 5×;功耗/面积高 | ✅ |
| dac | DAC_4 | JSSC 2021, 10.1109/jssc.2021.3057608 | 28 GS/s、8b、88 mW | 速率 2.8×;功耗略降 | ✅ |
| dac | DAC_5 | ESSCIRC 2022, 10.1109/esscirc55480.2022.9911426 | 56 GS/s、8b、280 mW | 速率 5.6×;功耗高 | ✅(可选) |
| tia | TIA_LP | MWSCAS 2016, 10.1109/MWSCAS.2016.7870132 | 1.95 mW、7 GHz | 功耗 −35% | ⚠️ **缺 area**;仿真结果 |
| tia | TIA_HB | JSSC 2012, 10.1109/JSSC.2011.2178723 | 9 mW、30 GHz | 带宽 3× | ⚠️ **缺 area** |
| mzm | TFLN_MZM_HL | Nat. Photonics 2019, 10.1038/s41566-019-0378-6 | IL 2.5、ER>40、112 Gbps | 速度/ER 优;eff ~700(估算)、宽缺失 | ⚠️ width 缺 |
| mzm | SOH_MZM | Sci. Rep. 2018, 10.1038/s41598-017-19061-8 | IL≈20、ER 14、<98 fJ/bit | 驱动能量优;IL 灾难 | ⚠️ width 缺 |
| mzm | Si_MZM_TWDD / TFLN_MZM_Hy110 | arXiv 1812.11081 / Sci. Rep. 2022, 10.1038/s41598-022-23403-6 | — | — | ❌ **缺必填键,崩溃** |
| photodetector | Sandia_GeWG_PD | OE 2011, 10.1364/OE.19.024897 | 1.3×4 µm、45 GHz、0.8 A/W | 紧凑+高速;灵敏度未知 | ✅ |
| photodetector | NEC_APD | Electron. Lett. 2006, 10.1049/el:20061700 | −30.2 dBm、20 GHz | 灵敏度优(全局激光预算);带宽低 | ⚠️ **缺面积/功耗** |
| phase_shifter | IHO_PS | LPR 2023, 10.1002/lpor.202201032 | 10×0.55 µm、9.6 mW、970 ns | 面积/功耗/响应优;IL 0.5 劣 | ✅ |
| phase_shifter | TFLN_EO_PS | arXiv 2304.03461 | 7000 µm、0.0006 mW、1.7 ns | P_pi/响应优;7 mm 长劣 | ⚠️ 预印本;P_pi 换算 |
| waveguide | WG_SiN_Strip_650nm | OE 2024, 10.1364/OE.514505 | 650 nm 宽、2.4 dB/cm | 可见-NIR 保守点 | ✅ |
| waveguide | WG_SiN_FurnaceAnnealed | LPR 2022, 10.1002/lpor.202200544 | 0.06 dB/cm | 超低损(**严格支配现有**) | ⚠️ 需决策;width 缺 |
| coupler | CPL_MMI_SiN_Angled_PS | OE 2020, 10.1364/OE.405188 | 80×13 µm、0.7 dB | 更大面积/IL;语义 PBS | ✅(语义需确认) |
| coupler | CPL_MMI_SiN_GMMI_2x2 | OE 2024, 10.1364/OE.514505 | IL<0.5 dB | 真 3-dB MMI | ⚠️ **缺几何** |
| y_branch | YJ_Si_Compact_1p2x2 | OE 2013, 10.1364/OE.21.001310 | 1.2×2 µm、0.28 dB | 实测替代理想 | ✅(硅工艺) |
| y_branch | SPL_SiN_GMMI_1x2 | OE 2024, 10.1364/OE.514505 | IL<0.5 dB | SiN 1x2 分束 | ⚠️ **缺几何** |
| micro_comb | MC_Si3N4_DKS_100GHz | Nature 2017, 10.1038/nature22387 | R 240 µm、FSR~100 GHz | 真实孤子微梳 | ✅ |
| mrr_rerouter | MRR_THERMAL | Photonics Res. 2019, 10.1364/PRJ.7.000155 | 10 mW/ring、IL 0.5 | IL 优;功耗 8-10 mW | ⚠️ **缺面积** |
| mrr_rerouter | MRR_PCM | Adv. Mater. 2020, 10.1002/adma.202001218 | 0 mW 保持 | 非易失零保持 | ⚠️ 定性 |

### 2.2 第二轮(6 类新器件,解锁 WDM/微环/PCM 族拓扑;调研方法复用第一轮)

| 新器件类型 | 锚定来源(DOI) | 服务拓扑 | 备注 |
|---|---|---|---|
| `mrr_weight`(微环权重调制器,2 变体) | Tait 2019, 10.1103/PhysRevApplied.11.064043;PR 2024, 10.1364/PRJ.520518 | broadcast_weight / mrr_crossbar / wdm_tensor_core | **区别于 mrr_rerouter(开关)** |
| `pcm_weight`(GST/GSST 相变权重单元) | Feldmann 2021, 10.1038/s41586-020-03070-1;Brückerhoff 2022, 10.1515/nanoph-2021-0752 | pcm_tensor_core | 非易失多态 |
| `wdm_mux`/`wdm_demux`(AWG 或级联 MZI) | Bai 2023, 10.1038/s41467-022-35506-9;PR 2024, 10.1364/PRJ.520518 | 所有 WDM 族 | — |
| `vcsel`(面发射激光器) | Chen 2023, 10.1038/s41566-023-01233-w | vcsel_coherent | — |
| `hybrid90`(90° 光混合器) | (Netcast/Youngblood 可选) | coherent_homodyne | 2×2 coupler 可近似 |
| `dispersive_delay`+`spectral_shaper` | Xu 2021, 10.1038/s41586-020-03063-0 | wdm_tensor_core(仅) | — |

## 3. 拓扑 ↔ 器件 耦合矩阵(两层如何结合)

| 拓扑(core_kind) | 复用库内器件 | 需新增器件 | 器件扩展覆盖? | 落地顺序 |
|---|---|---|---|---|
| `crossbar_dptc` | 全部现有 | 无 | ✅ 已覆盖(回填 Hamerly ref) | **第 0 步**(立即) |
| `mzi_mesh` / `butterfly_fft` / `mzi_mesh_reck` | PS/coupler/mmi/mzm/laser/PD/TIA/ADC/DAC/wg/crossing | 无(仅热光 PS 变体) | ✅ 基本够用 | **第 1 步**(第一轮器件扩展后) |
| `coherent_homodyne` | mzm/PD/TIA/ADC/DAC/coupler/laser | hybrid90(可选) | ✅ 基本够用 | **第 1 步** |
| `broadcast_weight` / `mrr_crossbar` | mzm/PS/PD/TIA/y_branch/mmi/crossing/laser | **mrr_weight + wdm** | ❌ 需第二轮 | **第 2 步** |
| `wdm_tensor_core` / `pcm_tensor_core` | micro_comb/PD/coupler/wg | **pcm_weight(或 mrr_weight)+ wdm + 色散** | ❌ 需第二轮 | **第 2 步** |
| `vcsel_coherent` | PD/TIA/ADC/coupler | **vcsel** | ❌ 需第二轮 | **第 2 步** |

**结合规则**:拓扑模板的 `devices` 表引用哪些类型,器件库就必须供给这些类型(否则 `build_sub_arch`/仿真器报错)。因此**拓扑扩展以器件扩展为前提**;MZI 族零新增可先行,WDM/微环/PCM 族必须等第二轮器件扩展。

## 4. 搜索空间修改

| 改动 | 内容 | 位置 |
|---|---|---|
| 新增拓扑模板维度 | `core_kind`(8 枚举)+ `node_kind`(3 枚举),白名单 | `search_space.json` + `hardware_ops.py` + `design_space.py` |
| 命名澄清 | 现 `topology` 层(共享/累加/激光分配)改名 `circuit`,新增 `topology`(结构层);或直接在现层下加 core_kind/node_kind | 同上 |
| 模板加载 | `build_sub_arch` 按 core_kind/node_kind 加载 `configs/design/{cores,nodes}/<kind>.yml` | `simulators/analytic.py` |
| 静态校验 | 加载模板时校验 DAG + 器件类型覆盖 + insertion_loss_final_device | `hardware_ops.py` |
| 证据探针 | `probe_topology`:枚举模板 + 结构诊断 | `evidence/probe.py` |
| 约束 | 能量公式覆盖的器件类型清单(新增 6 类需先加公式) | `onnarchsim/workflow/energy.py` |

## 5. agent 修改整理

| # | 对象 | 现状 | 修改 | 影响 |
|---|---|---|---|---|
| 1 | **topology agent 角色** | 只做损耗/功率预算诊断(固定拓扑) | **扩为两块**:① 结构选择(根因指向范式/并行度/数据流不匹配时,建议切换 core_kind/node_kind);② 预算诊断(现有) | `TopologyOutput` + `recommended_template`;`TOPOLOGY_PROMPT` |
| 2 | topology agent 工具 | summarize_topology / inspect_light_path(只读) | + `list_topology_templates` / `inspect_topology_template`(只读) | `builders.py` `TOPOLOGY_TOOLS` |
| 3 | 编辑工具 | modify/step_hardware_parameter + modify_device_parameter | + `modify_topology_template(core_kind, node_kind)`(白名单枚举) | solution 提议、orchestrator `_apply_edits` 确定性应用 |
| 4 | device agent | 选变体 | 在**选定模板约束下**选变体(模板缺器件时报错) | `DEVICE_PROMPT` |
| 5 | solution agent | 提议 actions/link_edits | 可提议 `modify_topology_template` | `SOLUTION_PROMPT` |
| 6 | **gate 接入循环** | 行 8 决策被丢弃(只写报告) | accept→采纳;request-info/re-execute→触发一次补证/重提;+ 结构合法性 deterministic 行 | `evaluate.py` + `orchestrator.py` |
| 7 | **终止条件** | 仅 max_iterations | + 无改进早停(patience)+ 收敛阈值(ε)+ 可选预算上限 | `orchestrator.py` |
| 8 | 证据闭环 | RAG 键无模板 | 指纹/检索描述加 core_kind/node_kind;probe_topology 结果入证据 | `rag/vector_store.py`、`evidence/` |

**职责硬边界(不变)**:拓扑模板是白名单枚举,LLM 只能选枚举、不得自由改网表;编辑由 orchestrator 确定性应用(与现有 edit 工具同机制)。

## 6. 落地顺序与待决问题

**多轮扩展清单(2025-10-21 重定,逐轮审批)**:

| 轮次 | 方面 | 内容 | 状态 |
|---|---|---|---|
| **第 0 轮** | 基线打底 | 回填 DPTC 的 Hamerly ref(10.1103/PhysRevX.9.021032)+ 搜索空间快照一致性(补 mmi/off_chip_laser)+ 修复 verify 脚本 import | ✅ 已完成(94 测试绿) |
| **第 1 轮** | 器件扩展·**缩窄版** | **只扩"权衡轴 + 拓扑急需依赖"的 4 类**:adc×1(ADC_SAR_90GS)、dac×2(DAC_3/DAC_4)、phase_shifter×1(IHO_PS)、photodetector×1(Sandia_GeWG_PD);全部带 `platform` 标注;tia/waveguide/coupler/y_branch/mrr_rerouter/mzm **搁置**(理由见 round1 清单 B 组) | ✅ **已完成**(94 测试绿、快照一致、冒烟通过) |
| **第 2 轮** | 拓扑扩展·第一批 + 搜索空间 + agent | 零新增器件的拓扑:`mzi_mesh`(Clements/Reck)、`butterfly_fft`、`coherent_homodyne`(OCDC/Netcast)+ `node_kind`(`tunable_mzi`);搜索空间加 `core_kind`/`node_kind` 白名单维度 + 模板加载 + 静态校验 + probe_topology;agent:topology 角色扩为"结构选择+预算诊断"、`modify_topology_template` 工具、gate 接入循环、终止条件、证据闭环 | 待审批(依赖第 1 轮:phase_shifter/coupler/pd/adc/dac) |
| **第 3 轮** | 器件扩展·第二批 + 拓扑扩展·第二批 | 6 类新器件(mrr_weight/pcm_weight/wdm_mux·demux/vcsel/hybrid90/dispersive_delay+spectral_shaper)+ WDM/微环/PCM 族 5 类拓扑 + energy 公式扩展 + 平台一致性/集成成本约束 | 待审批 |

**待决(需你拍板)**:① 第 1 轮 A 组 5+2 个是否确认入库;② ADC_SAR_90GS 的 out_bit 上限是否限 6;③ `platform` 标注是否接受(约束后置);④ topology 层是否改名 circuit(第 2 轮);⑤ 第二轮器件扩展是否立项(第 3 轮);⑥ gate 接入力度与早停参数默认值(第 2 轮);⑦ 接受"LLM 白名单枚举、不得自由改网表"边界(第 2 轮);⑧ MZI 网格空间模式数 M 的映射方式(第 2 轮)。
