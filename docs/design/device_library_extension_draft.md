# 器件库扩展草案:帕累托有效的真实器件候选清单

> 状态:**草案待审阅** — 未写入 `configs/devices/`。本文列出的每个候选都必须经过审阅确认后才可入库。
> 关联分析:器件层搜索有效性分析(见本目录对话结论);上游来源 `ScopeX-ASU/SimPhony`(vendored 于 `photon_tuner/simulators/simphony/`)。

## 0. 变更记录

| 日期 | 变更 | 位置 |
|---|---|---|
| 2025-10-21 | **理想器件从器件搜索空间移除**(按本对话决策):`Customized_PS`(P_pi=0/resp=0)、片上 `LT_Laser`(0 dB IL)、`Customized_Y_Branch`(10 nm 宽占位)、`Customized_DC_1x1`(2x2 槽位中的 1x1 光栅)。实现方式:发现层过滤(`EXCLUDED_SEARCH_VARIANTS`,见 §5.2),YAML 库与仿真器 readback 保持不动;同步更新 `search_space.json` 快照。94 个测试全过。 | `photon_tuner/core/search_space.py`、`configs/search_space.json` |

## 1. 目标与约束

- **目标**:让器件层搜索在"真实有效"的意义上覆盖更多维度——即每个可搜索类型至少存在 2 个**帕累托有效**(互不支配、有真实权衡)的候选,而不是"候选多但被理想器件/支配器件淹没"。
- **扩展方式**:与 SimPhony 完全相同——在 `configs/devices/{photonics,electronics}/*.yml` 中按既有 schema 增加变体(YAML 加载机制、运行时发现、仿真器消费全部自动生效,零代码改动)。
- **来源约束**:每个新条目必须对应一篇**已发表的真实器件文献**,`ref` 填可解析的 DOI/arXiv 链接;参数值必须来自论文报道,禁止臆造;未报道的参数**省略**并标注,不猜。
- **有效性约束(帕累托规则)**:新条目必须相对现有变体在 ≥1 个轴上更优、≥1 个轴上更差(真实权衡)。严格支配现有变体、或被现有变体严格支配的条目**不收录**(否则搜索再次退化)。
- **schema 约束**(仿真器实际消费的键,`onnarchsim/workflow/{energy,area,insertion_loss}.py` + `utils.py:350-357`):
  - 面积 = `cfgs.area` 或 `cfgs.width × cfgs.length`(二者必居其一)
  - adc:`type ∈ {sar, flash}`、`sample_rate`(GSPS)、`prec`(bit)、`power`(mW 基准);可行域 `out_bit ≤ prec`、`work_freq ≤ sample_rate`
  - dac:`prec`、`sample_rate`、`power`、可选 `FoM`(非空则 P = 2·FoM·in_bit·wf·1e-3);可行域 `in_bit ≤ prec`、`work_freq ≤ sample_rate`
  - tia:`area`、`dynamic_power`(直接消费)
  - mzm:`length/width`、`insertion_loss`(dB)、`extinction_ratio`(dB)、`efficiency`(fJ/bit)、`testing_bit`、`modulation_speed`(Gbps)、`eo_bandwidth`;可行域 `work_freq ≤ modulation_speed/testing_bit`
  - phase_shifter:`length/width`、`insertion_loss`、`P_pi`(mW)、`response_time`(ns;tuning cycles = resp × wf,直接进时延)
  - photodetector:`length/width`、`sensitivity`(dBm,全芯片取最不灵敏者进激光预算)、`bandwidth`(GHz,work_freq 上限输入)、`responsivity`;功耗用 `static_power`/`dynamic_power`
  - waveguide:`mode`、`width`(nm)、`type`、`insertion_loss`(dB/cm)
  - coupler / y_branch / laser_splitter:`width`、`length`、`insertion_loss`
  - micro_comb:`width`(nm)、`length`(um)仅面积
  - mrr_rerouter:`static_power`、`length`、`width`、`insertion_loss`
- **目标函数影响**(latency × 每MAC能量 × 面积):
  - IL 每 1 dB 使激光能量 ×1.26(指数级,`cal_laser_power`);ER 通过 extinction_scale = 1/(1−10^(−ER/10)) 乘激光功率;sensitivity 全局生效(最差者定预算)
  - MZM 动态功耗 = eff × wf × 1e-3 × testing_bit;PS 动态功耗 = P_pi;时延 ∝ max tuning_cycles
  - 当前瓶颈:ADC `sample_rate=10 GSPS` 把 work_freq 上限压到 10 GHz → 高速 ADC 是解锁 work_freq 15-25(降时延)的关键轴

## 2. 现有库的有效性盘点(写入前的基准)

| 类型 | 候选数 | 有效(Pareto)候选 | 无效原因 |
|---|---|---|---|
| mzm | 4 | LT_MZM, SL_MZM | ACADEMIA_1x1/2x2 被 LT 支配(除 ER 外的所有轴) |
| dac | 2 | DAC_1, DAC_2 | —(按精度真实权衡) |
| photodetector | 3 | ACADEMIA_Digital_PD, LT_PD(弱) | Analog_PD 被 Digital 支配;LT_PD 功耗/灵敏度劣 |
| phase_shifter | 4 | Customized_PS(理想器件) | Customized_PS(P_pi=0, resp=0, 无 ref)支配其余;ITO 仅 IL 微弱反超 |
| coupler | 4 | LT_DC_2x2 | ACADEMIA/Customized_2x2 被 LT 支配;1x1 光栅槽位物理不合法 |
| y_branch | 2 | Customized(理想) | 支配 LT |
| on_chip_laser | 2 | LT_Laser | 0 IL(理想化)支配 Customized(IL 2) |
| adc/tia/waveguide/laser_splitter/micro_comb/mrr_rerouter | 各 1 | — | 无搜索 |

有效决策点合计 ≈ 5(MZM、DAC、PD、laser_source 开关、splitter_kind 开关)。

> **§0 变更后(理想器件已移出搜索面)**:phase_shifter 剩 ITO/MEMS/TO 三个真实候选——ITO(IL 0.03, 15 mW, 10000 ns, 500 µm²)与 MEMS(0.33, 0 mW, 2000 ns, 4500 µm²)构成真实 IL/面积 vs 功耗/时延权衡,该维度开始有效;on_chip_laser 只剩 Customized_Laser(IL 2),与 off-chip LT_Laser(IL 2, wpe 0.3)的 on/off 开关成为纯效率权衡;y_branch 只剩 LT_Y_Branch(单候选,待补真实条目);coupler 剩 3 个但 LT 仍支配其余。

## 3. 调研候选清单(8 路调研已完成;每条均附 DOI/arXiv 与来源质量声明)

### 3.0 器件总览:名称 / 中文名 / 作用 / 库中是否原有 / 待扩展

| 名称 | 中文名 | 作用(在光加速器中的角色) | 库中是否原有 | 待扩展(调研候选,附来源) |
|---|---|---|---|---|
| **mzm** | 马赫-曾德尔调制器 | 将输入/权重电信号调制到光载波——光点积的乘法执行器件 | 原有(4 变体) | TFLN_MZM_HL(Nat. Photonics 2019)、SOH_MZM(Sci. Rep. 2018);另 2 个缺必填键不可用 |
| **dac** | 数模转换器 | 数字输入 → 驱动调制器的模拟信号 | 原有(2) | DAC_3(SSCL 2022)、DAC_4(JSSC 2021)、DAC_5(ESSCIRC 2022,可选) |
| **adc** | 模数转换器 | 光电探测结果 → 数字读出(当前 10 GS/s 是 work_freq 瓶颈) | 原有(1) | ADC_SAR_90GS(ISSCC 2014,解锁 15-25 GHz)、ADC_SAR_4P8GS_LP(JSSC 2026) |
| **tia** | 跨阻放大器 | 光电探测器后的电流→电压放大 | 原有(1) | TIA_LP(MWSCAS 2016)、TIA_HB(JSSC 2012) |
| **waveguide** | 光波导 | 器件间光传输介质(损耗 dB/cm) | 原有(1) | WG_SiN_Strip_650nm(OE 2024)、WG_SiN_FurnaceAnnealed(LPR 2022) |
| **laser_splitter** | 激光分束器 | 激光功率扇出到多波长通道 | 原有(1) | —(暂无候选) |
| **mmi** | 多模干涉分束器 | 1xN 分束/合束(亦可作 2x2 耦合器) | 原有(1) | CPL_MMI_SiN_Angled_PS(OE 2020)、CPL_MMI_SiN_GMMI_2x2(OE 2024) |
| **micro_comb** | 微梳光源 | 多波长光频梳,替代多激光器 | 原有(1) | MC_Si3N4_DKS_100GHz(Nature 2017,真实孤子微梳) |
| **mrr_rerouter** | 微环重路由开关 | 环形谐振器做信号路由/重排 | 原有(1) | MRR_THERMAL(Photonics Res. 2019)、MRR_PCM(Adv. Mater. 2020) |
| **on_chip_laser** | 片上激光器 | 片上光功率源(理想 LT_Laser 已移出搜索面) | 原有(2→1) | — |
| **off_chip_laser** | 片外激光器 | 封装光功率源(效率更高,本地微调条目) | 原有(1) | — |
| **phase_shifter** | 相移器 | 施加权重相移——光点积的核心权重器件 | 原有(4→3,理想条目已移除) | IHO_PS(Laser & Photonics Rev. 2023)、TFLN_EO_PS(arXiv 2023) |
| **photodetector** | 光电探测器 | 光信号 → 电流(灵敏度决定全局激光预算) | 原有(3) | NEC_APD(Electron. Lett. 2006)、Sandia_GeWG_PD(Opt. Express 2011) |
| **coupler** | 耦合器 | 2x2 分束/合束,构成 MZI 的基本单元 | 原有(4→3,1x1 已移除) | CPL_MMI_SiN_Angled_PS(OE 2020)、CPL_MMI_SiN_GMMI_2x2(OE 2024) |
| **y_branch** | Y 分支 | 1x2 分束(理想条目已移除,现仅 1 个) | 原有(2→1) | YJ_Si_Compact_1p2x2(OE 2013)、SPL_SiN_GMMI_1x2(OE 2024) |
| crossing | 波导交叉 | 交叉走线(库内但不在网表,不参与搜索) | 原有(5) | — |
| large_insertion | 大插损测试件 | 测试/占位(不在网表) | 原有(1) | — |
| integrator | 积分器 | 电子时间积分(不在网表) | 原有(1) | — |

**候选可入库状态**:✅ 可直接入库 = TFLN_MZM_HL¹、DAC_3、DAC_4、DAC_5、ADC_SAR_90GS¹、Sandia_GeWG_PD、IHO_PS、WG_SiN_Strip_650nm、CPL_MMI_SiN_Angled_PS²、YJ_Si_Compact_1p2x2、MC_Si3N4_DKS_100GHz。⚠️ 需补参数/先决策(缺面积、width 缺失致 area=0、估算值、支配式扩展、预印本等,见 §3.1-3.8 各节标注)。❌ 不可用(缺 mzm 必填键 → 仿真器崩溃):Si_MZM_TWDD、TFLN_MZM_Hy110。
¹ 需保留"估算/未报道"注释;² 语义为偏振分束器(PBS),作 coupler 需确认。

> **阅读指引**:上表是全部器件的一页速览。§3.1–3.8 按类别给出每个候选的**论文 DOI、逐条参数、相对现有变体的权衡轴、YAML 示例与来源质量声明**——审阅候选时看对应小节;§4 是每条入库前必须逐项打勾的清单。

### 3.1 adc(解锁 work_freq 的关键轴)
> 背景核实:现有 `ADC_SAR_1` = JSSC 2022 "A 10-GS/s 8-bit 2850-µm² Two-Step Time-Domain ADC…"(10.1109/jssc.2022.3196924,10 GS/s、~14.8 mW、2850 µm²、14nm)。调研以 Murmann ADC survey 1997-2026 + Crossref/OpenAlex 全文交叉核对;原始 survey 存于 `survey_sheet2.tsv`。

**候选 1:ADC_SAR_90GS(高速,解锁 15-25 GHz)** — 90GS/s_8b_667mW
- 论文:A 90GS/s 8b 667mW 64× Interleaved SAR ADC in 32nm Digital SOI CMOS(ISSCC 2014, pp. 378-379)
- DOI: https://doi.org/10.1109/ISSCC.2014.6757477
- 参数:type sar, sample_rate 90 GSPS(实测至 100 GS/s), prec 8, power 667 mW, area 450000 µm²(0.45 mm²), tech 32 nm
- 权衡轴 vs ADC_SAR_1:速率 90 > 10(解锁 work_freq 15-25 → 降时延);代价 power 667 ≫ 14.8(×45)、area 450000 ≫ 2850(×158)→ 真实帕累托点(速率轴)
- 支配检查:不被支配(≥16 GS/s 8bit ADC 中无功率/面积更优者)
- 可行性:SNDR 33-36 dB ⇒ ENOB ≈ 5.2-5.7;仿真器只检查 out_bit ≤ prec(8),但**物理上 out_bit=8 偏乐观**,建议搜索空间内用 out_bit ≤ 6 或标注;tech 32nm ≠ 库 14nm(PDK 空即可)
```yaml
adc:
  ADC_SAR_90GS:
    PDK: ""
    ref: "https://doi.org/10.1109/ISSCC.2014.6757477"
    type: ADC
    exp: True
    cfgs:
      type: sar
      sample_rate: 90 # GSPS
      prec: 8 # bit (SNDR 33-36 dB -> ENOB ~5.2-5.7; nominal 8b as reported)
      power: 667 # mW
      area: 450000 # um2 (0.45 mm2)
      tech: 32 # nm
```

**候选 2:ADC_SAR_4P8GS_LP(低功耗)** — 4.8GS/s_8b_10.4mW
- 论文:An 8-bit 4.8-GS/s Four-Channel Time-Interleaved SAR ADC With Ping-Pong Comparators…(JSSC, vol. 61, 2026, 清华)
- DOI: https://doi.org/10.1109/jssc.2026.3677867
- 参数:type sar, sample_rate 4.8 GSPS, prec 8, power 10.4 mW(含 buffer 与校准), tech 28 nm,**area 未报道**
- 权衡轴 vs ADC_SAR_1:power 10.4 < 14.8(−30%);代价 sample_rate 4.8 < 10(work_freq 上限 10→4.8,反而更紧)→ 真实帕累托点(功耗轴),但**速率代价为负**
- 支配检查:不被支配(5-10 GS/s 区间内唯一功耗更优的真实 8bit SAR;≥5 GS/s 的 8bit SAR 均为功耗支配:29 mW@5GS/s、138.6 mW@5GS/s、57.3 mW@10GS/s)
- ⚠️ **area 未报道 → 仿真器抛错(`utils.py:354-358` 要求 area 或 width×length 必居其一),入库前必须补面积来源或换候选**
- 可行性:SNDR 44.3 dB ⇒ ENOB ≈ 7.1(8bit 真实可用);work_freq ≤ 4.8 GHz
```yaml
adc:
  ADC_SAR_4P8GS_LP:
    PDK: ""
    ref: "https://doi.org/10.1109/jssc.2026.3677867"
    type: ADC
    exp: True
    cfgs:
      type: sar
      sample_rate: 4.8 # GSPS
      prec: 8 # bit (SNDR 44.3 dB -> ENOB ~7.1)
      power: 10.4 # mW
      tech: 28 # nm
      # area: not reported in the paper — MUST be resolved before library entry
```

**调研结论**:① ADC_SAR_90GS 是解锁 15-25 GHz 的真实候选(代价 ×45 功耗/×158 面积);② 低功耗槽位**没有 ≥5 GS/s 且功耗优于 ADC_SAR_1 的真实 8bit SAR**——ADC_SAR_1 本身就在前沿上;唯一低于 14.8 mW 的是 4.8 GS/s(JSSC 2026)。

### 3.2 dac
**候选 1:DAC_3(低 FoM)** — `Intel16_9b_50GSps`
- 论文:A 48-dB SFDR, 43-dB SNDR, 50-GS/s 9-b 2x-Interleaved Nyquist DAC in Intel 16(IEEE SSCL, 2022)
- DOI: https://doi.org/10.1109/lssc.2022.3205884
- 参数:prec 9, sample_rate 50 GSPS, power 243 mW, area 200000 um²(0.2 mm² active), tech 16(Intel 16,约 16nm 级 FinFET), FoM 23.2 fJ/step(论文同时报道 41.3 SNDR FoM)
- 权衡轴 vs DAC_1:FoM 23.2 < 33.6、速率 50 > 14、每采样能量 4.86 pJ < 12.07 pJ;代价 power 243 > 169、area 200000 > 11000、prec 9 < 12 → **真实帕累托点**
- vs DAC_2:prec 9 > 8、速率 50 > 14;代价 power/area → 真实帕累托点
- 支配检查:均不支配;仿真器走 FoM 路径(P = 2·23.2·in_bit·wf·1e-3),243 mW 基值被覆盖
- 可行性:in_bit ≤ 9;work_freq ≤ 50(上限 14→50);注意实测 ENOB 约 6.9-7.7 bit(prec 9 略虚标)
```yaml
dac:
  DAC_3:
    PDK: ""
    ref: "https://doi.org/10.1109/lssc.2022.3205884"
    type: DAC
    exp: True
    cfgs:
      tech: 16 # Intel 16 (~16nm-class FinFET; paper reports process name only)
      area: 200000 # um^2 (0.2 mm2 active)
      prec: 9 # bit
      power: 243 # mW (base; simulator recomputes via FoM path)
      sample_rate: 50 # GSPS
      FoM: 23.2 # fJ/step (linearity FoM; SNDR FoM 41.3 also reported)
```

**候选 2:DAC_4(高速)** — `16nm_8b_28GSps`
- 论文:A 2x Time-Interleaved 28-GS/s 8-Bit 0.03-mm² Switched-Capacitor DAC in 16-nm FinFET CMOS(JSSC, 2021)
- DOI: https://doi.org/10.1109/jssc.2021.3057608
- 参数:prec 8, sample_rate 28 GSPS, power 88 mW, area 30000 um², tech 16, FoM null(未报道)
- 权衡轴 vs DAC_2:速率 28 > 14(2×),代价 power 88 > 50、area 30000 > 11000,同 8bit/16nm → 真实帕累托点
- vs DAC_1:速率与功耗双优(28 > 14、88 < 169),prec 8 < 12、area 更大 → 真实帕累托点
- 可行性:in_bit ≤ 8;work_freq ≤ 28;FoM null → 仿真器用基值 88 mW
```yaml
dac:
  DAC_4:
    PDK: ""
    ref: "https://doi.org/10.1109/jssc.2021.3057608"
    type: DAC
    exp: True
    cfgs:
      tech: 16 # nm FinFET
      area: 30000 # um^2 (0.03 mm2)
      prec: 8 # bit
      power: 88 # mW
      sample_rate: 28 # GSPS
      FoM: null # not reported
```

**候选 3(可选):DAC_5(极限速率)** — `16nm_8b_56GSps`
- 论文:A 56 GS/s 8-bit 0.011 mm² 4x Delta-Interleaved Switched-Capacitor DAC in 16nm FinFET CMOS(ESSCIRC, 2022)
- DOI: https://doi.org/10.1109/esscirc55480.2022.9911426
- 参数:prec 8, sample_rate 56 GSPS, power 280 mW, area 11000 um²(与 DAC_1/2 相同), FoM null
- 权衡轴:速率 4× DAC_2(56 vs 14)换 power 5.6×(280 vs 50),面积零变化 → 纯"速率上限 ×4"轴
```yaml
dac:
  DAC_5:
    PDK: ""
    ref: "https://doi.org/10.1109/esscirc55480.2022.9911426"
    type: DAC
    exp: True
    cfgs:
      tech: 16 # nm FinFET
      area: 11000 # um^2 (0.011 mm2)
      prec: 8 # bit
      power: 280 # mW
      sample_rate: 56 # GSPS
      FoM: null # not reported
```
**调研备注**:另核实的 8b 100 GS/s 分布式 DAC(TMTT 2015)因 ENOB 仅 3.2-5.3 bit、2.5 W 被拒;25 GS/s 8b 40nm TVLSI 2025 被 DAC_4 支配。建议收录 DAC_3 + DAC_4(DAC_5 可选)。

### 3.3 tia
> ⚠️ **关键缺口**:仿真器 `load_device_and_count`(`utils.py:354-358`)要求器件必须有 `area` 或 `width×length`,否则直接抛错。下方两个候选的论文均**未报告 TIA 核面积**——入库前必须:① 另找报道面积/版图的 TIA 论文,或 ② 用论文中的裸片面积并明确标注为"工程估算、非论文参数"(违反"参数必须来自论文"原则,需你批准),或 ③ 放弃该候选。

**候选 1:TIA_LP(低功耗,仿真级)** — 130nm_7GHz_1.95mW
- 论文:10 Gb/s 1.95 mW active cascode transimpedance amplifier for high speed optical receivers(MWSCAS 2016)
- DOI: https://doi.org/10.1109/MWSCAS.2016.7870132
- 参数:dynamic_power 1.95 mW(1.5V),带宽 7 GHz,@0.2pF PD;tech 130nm;增益 56.65 dBΩ;sensitivity −20.5 dBm@BER 1e-12;area 未报道
- ⚠️ 数值为**版图后仿真**,非实测硅片
- 权衡轴 vs TIA_1:功耗 1.95 < 3 mW(−35%);带宽 7 GHz 低、tech 130nm 老 → 真实帕累托点(功耗轴)
- YAML(cfgs 仅 dynamic_power;area 缺 → **不可直接入库**)
```yaml
tia:
  TIA_LP:
    PDK: ""
    ref: "https://doi.org/10.1109/MWSCAS.2016.7870132"
    type: TIA
    exp: True # NOTE: post-layout simulation only (not measured silicon)
    cfgs:
      dynamic_power: 1.95 # mW
```

**候选 2:TIA_HB(高带宽,实测)** — 45nmSOI_30GHz_9mW
- 论文:A 40-Gb/s Optical Transceiver Front-End in 45 nm SOI CMOS(JSSC, 2012)
- DOI: https://doi.org/10.1109/JSSC.2011.2178723(姊妹篇 CICC 2010: 10.1109/CICC.2010.5617418 数值一致)
- 参数:带宽 30 GHz(TIA+后放,55 dBΩ)、dynamic_power 9 mW(TIA,1V)、tech 45nm SOI、噪声 20.5 pA/√Hz;area 未报道
- 权衡轴 vs TIA_1:带宽 30 GHz ≫、功耗 9 mW(3×)→ 真实帕累托点(带宽轴)
```yaml
tia:
  TIA_HB:
    PDK: ""
    ref: "https://doi.org/10.1109/JSSC.2011.2178723"
    type: TIA
    exp: True # measured silicon (45 nm SOI)
    cfgs:
      dynamic_power: 9 # mW (TIA, 1V)
```

**调研备注**:备选 NTT 56-Gb/s TIA(CSICS 2016, 38.4 GHz 实测带宽)摘要未报功耗;100 Gb/s 65nm TIA(ISCAS 2014, 70 GHz, 24 mW)为仿真结果。两者同样缺面积。**该类别入库前必须解决 area 来源问题。**

### 3.4 mzm
> 仿真器对 mzm 的必读键(缺失即崩溃):`extinction_ratio`(`energy.py:161-165` 直接下标访问)、`efficiency`(`cal_MZM_power`)、`modulation_speed`(`max_symbol_rate`)。**凡缺这三者之一的候选不可直接入库。** `length`/`width` 缺一 → area=0(不崩溃,但物理失真)。

**候选 1:TFLN_MZM_HL(混合 Si-LN,TFLN 级)** — 112Gbps_ER>40dB
- 论文:High-performance hybrid silicon and lithium niobate Mach–Zehnder modulators for 100 Gbit s⁻¹ and beyond(Nature Photonics 13, 2019)
- DOI: https://doi.org/10.1038/s41566-019-0378-6(arXiv: 1807.10362)
- 参数:length 3000 µm(3-mm 臂,另有 5-mm 器件), width 未报道, insertion_loss 2.5 dB, extinction_ratio >40 dB(静态;56/72/84/100 Gb/s 动态 ER 11.8/6.0/5.5/5.0), efficiency ~700 fJ/bit(**作者估算**,非直接实测), testing_bit 1, eo_bandwidth >70 GHz, modulation_speed 112 Gbps(56 GBaud PAM-4;100 Gb/s OOK), Vπ·L 2.2 V·cm
- 权衡轴 vs LT/SL:速度 112 > 25、带宽 >70 > 12.5、ER >40 > 10/6 全优;代价 IL 2.5 > LT 的 1.2(劣于 SL 4.5)、eff 700 > 450/50、长度 3000 µm ≫ → **真实帕累托点(速度/ER 换损耗与能量)**
- ⚠️ width 未报道 → area=0(失真);eff 为估算值,需标注
```yaml
mzm:
  TFLN_MZM_HL:
    PDK: ""
    ref: "https://doi.org/10.1038/s41566-019-0378-6"
    type: MZM
    exp: True
    cfgs:
      length: 3000 # um (3-mm phase-shifter arm)
      # width: not reported -> area=0 caveat
      insertion_loss: 2.5 # dB
      extinction_ratio: 40 # dB (static; dynamic 11.8@56Gb/s ... 5.0@100Gb/s)
      efficiency: 700 # fJ/bit (author estimate, not directly measured)
      testing_bit: 1
      eo_bandwidth: 70 # GHz (>70, beyond VNA limit)
      modulation_speed: 112 # Gbps (56 GBaud PAM-4)
```

**候选 2:SOH_MZM(硅-有机混合,极低驱动能量)** — 100Gbps_<98fJ/bit
- 论文:Silicon-Organic Hybrid (SOH) Mach-Zehnder Modulators for 100 Gbit/s on-off Keying(Scientific Reports, 2018)
- DOI: https://doi.org/10.1038/s41598-017-19061-8(开放获取)
- 参数:length 1100 µm(1.1-mm slot 相移器), width 未报道, insertion_loss ≈20 dB(论文损耗预算加总:9 dB 光栅对 + ~1+1 dB 转换器/MMI + ~1 dB 波导 + ≈8 dB slot 段), extinction_ratio 14 dB(静态;动态 5-7@100Gb/s), efficiency <98 fJ/bit, testing_bit 1, eo_bandwidth 25 GHz(6-dB EOE 点), modulation_speed 100 Gbps OOK, Vπ 0.9 V
- 权衡轴 vs LT/SL:速度 100 > 25、eff 98 < LT 450(但 > SL 50)、ER 14 > 10/6;代价 **IL ≈20 dB**(相对 LT 1.2 是灾难性损耗 → 激光能量 ×~63×10^1.88)、长度 1100 µm → **真实帕累托点(驱动能量/速度换损耗)**,但 IL 预算惩罚下几乎不会胜出
- ⚠️ IL 20 dB 是论文预算分项加总,非论文单一报道值;width 缺失 → area=0
```yaml
mzm:
  SOH_MZM:
    PDK: ""
    ref: "https://doi.org/10.1038/s41598-017-19061-8"
    type: MZM
    exp: True
    cfgs:
      length: 1100 # um (1.1-mm slot phase shifter)
      # width: not reported -> area=0 caveat
      insertion_loss: 20 # dB (sum of paper's stated loss-budget components)
      extinction_ratio: 14 # dB (static; dynamic 5-7 at 100 Gb/s)
      efficiency: 98 # fJ/bit (<98, record-low for 100G OOK)
      testing_bit: 1
      eo_bandwidth: 25 # GHz (6-dB EOE point)
      modulation_speed: 100 # Gbps (OOK, 10 km)
```

**候选 3:Si_MZM_TWDD(硅行波双驱动)— ⚠️ 缺必填键,不可直接入库**
- 论文:Up to 168-Gb/s PAM-4 direct detection transmission with silicon travelling wave MZM(arXiv, 2018;器件细节见 arXiv:1811.11096)
- arXiv: https://arxiv.org/abs/1812.11081(仅预印本,无期刊版)
- 参数:length 2500 µm, eo_bandwidth 21.4 GHz(3-V 反偏;0 V 时 14.7), modulation_speed 168 Gbps(84 GBaud PAM-4), testing_bit 2,**IL/ER/efficiency 均未报道**
- ⚠️ **ER/efficiency 缺失 → 仿真器 KeyError 崩溃;两键均为 mzm 必读**。需从全文补数后才能考虑,当前列为"线索"而非候选。

**候选 4:TFLN_MZM_Hy110(110 GHz 极速)— ⚠️ 同样缺必填键**
- 论文:110 GHz, 110 mW hybrid silicon-lithium niobate Mach-Zehnder modulator(Sci. Rep., 2022, 开放获取)
- DOI: https://doi.org/10.1038/s41598-022-23403-6
- 参数:length 17000 µm(总长 ~1.7 cm), insertion_loss 1.8 dB(片上), extinction_ratio 28 dB(50 nm 均值;max 31), eo_bandwidth >110 GHz, Vπ·L 3.1 V·cm;**efficiency / testing_bit / modulation_speed 未报道**(论文为模拟/RF 光子学应用,无数字传输测试)
- ⚠️ 三个必读键缺失 → 不可直接入库。

**调研备注**:① DOI 勘误:Wang 2018 Nature TFLN 调制器真实 DOI 为 10.1038/s41586-018-0551-y(非 10.1038/s41586-018-0824-9);"High-performance hybrid lithium niobate" 是 He et al. Nat. Photonics 13, 359 (2019),非 Nature 2018。② Wang 2018 是经典候选(210 Gbit/s、片上损耗 <0.5 dB、CMOS 兼容驱动),但 Vπ/带宽/ER/面积在付费全文内 → 标注未报道,需有权限者补。③ Optica 系全文(Photon. Res. 2018、OE 2017 等)被反爬保护,未核实数字一律未引用。

### 3.5 photodetector
**候选 1:NEC_APD(高灵敏度,实测模块级)** — 20GHz_−30.2dBm
- 论文:10 Gbit/s asymmetric waveguide APD with high sensitivity of −30 dBm(Electronics Letters 42(20), 2006, NEC)
- DOI: https://doi.org/10.1049/el:20061700
- 参数:bandwidth 20 GHz, sensitivity **−30.2 dBm**@10 Gb/s(APD+TIA 接收模块级),增益带宽积 110 GHz,QE 94%@1.55µm;**area/length/width、responsivity、功耗未报道**
- 权衡轴 vs ACADEMIA_Digital_PD(−27 dBm,27 GHz,320 µm²):灵敏度 −30.2 优 3.2 dB(**全局激光预算 ×~0.48**);带宽 20 < 27(work_freq 上限 27→20);面积/功耗未验证 → 真实帕累托点(灵敏度轴)
- ⚠️ 面积缺失 → 仿真器抛错,入库前必须解决
```yaml
photodetector:
  NEC_APD:
    PDK: ""
    ref: "https://doi.org/10.1049/el:20061700"
    type: Photodetector
    exp: True
    cfgs:
      bandwidth: 20 # GHz
      sensitivity: -30.2 # dBm (receiver-module level, 10 Gb/s BER)
      # length/width/power not reported — MUST be resolved before entry
```

**候选 2:Sandia_GeWG_PD(紧凑高带宽,实测)** — 45GHz_5.2um²
- 论文:Ultra compact 45 GHz CMOS compatible Germanium waveguide photodiode with low dark current(Optics Express 19, 24897, 2011, Sandia)
- DOI: https://doi.org/10.1364/OE.19.024897
- 参数:footprint 1.3×4 µm²(=5.2 µm²), bandwidth 45 GHz, responsivity 0.8 A/W, dark current 3 nA;**sensitivity/功耗未报道**
- 权衡轴 vs ACADEMIA_Digital_PD:面积 5.2 ≪ 320(×60 小)、带宽 45 > 27 双优;代价 responsivity 0.8 < 1.1、灵敏度未知(无法与 −27 比较)→ 真实帕累托点(面积/带宽轴)
```yaml
photodetector:
  Sandia_GeWG_PD:
    PDK: ""
    ref: "https://doi.org/10.1364/OE.19.024897"
    type: Photodetector
    exp: True
    cfgs:
      length: 4 # um
      width: 1.3 # um
      responsivity: 0.8 # A/W
      bandwidth: 45 # GHz
      # sensitivity/power not reported
```

**调研备注**:">=28 GHz 且 ≤−28 dBm"组合处于文献帕累托边缘;核实过的前沿点:NEC EL 2006 是唯一同时明确报道 20 GHz 带宽与 ≤−28 dBm 灵敏度的真实 PD;其他(OE 2022 48GHz APD −21.3 dBm、OL 2021 Ge APD −21 dBm、NTT 50G APD −10.8 dBm)灵敏度均不足。注意:所有灵敏度均为接收模块级(APD+TIA、BER 阈值),非裸 PD 参数。

### 3.6 phase_shifter
> 结论前置:两个真实候选相对 **ITO_PS / MEMS_PS** 都是帕累托有效(各赢 ≥1 轴);但两者都仍被理想条目 **Customized_PS**(IL 0.05、P_pi 0、resp 0、无 ref)支配——**在解决 Customized_PS 之前,PS 的搜索依然退化**。只有 IHO 候选在面积轴也胜 Customized_PS。

**候选 1:IHO_PS(掺氢氧化铟热光相移器)** — 10um_9.6mW_970ns
- 论文:An Efficient, Fast-Responding, Low-Loss Thermo-Optic Phase Shifter Based on a Hydrogen-Doped Indium Oxide Microheater(Laser & Photonics Reviews, 2023)
- DOI: https://doi.org/10.1002/lpor.202201032(arXiv: 2301.00816)
- 参数:length 10 µm, width 0.55 µm(550nm 条波导), insertion_loss 0.5 dB(微加热器附加损耗), P_pi 9.6 mW, response_time 970 ns(rise;fall 980 ns)
- 权衡轴 vs ITO_PS(50×10, IL 0.03, 15 mW, 10000 ns):长度/宽度/P_pi/响应 4 轴更优,IL 0.5 劣;vs MEMS_PS(100×45, 0.33, 0, 2000 ns):长度/宽度/响应 3 轴更优,IL/P_pi 劣 → **对两个真实器件均帕累托有效**
- 对 Customized_PS:仅面积轴胜(10×0.55 vs 33×6.5),P_pi/IL/resp 全劣
```yaml
phase_shifter:
  IHO_PS:
    PDK: ""
    ref: "https://doi.org/10.1002/lpor.202201032"
    type: PhaseShifter
    exp: True
    cfgs:
      length: 10 # um (IHO heater film along propagation)
      width: 0.55 # um (550-nm strip waveguide)
      insertion_loss: 0.5 # dB (excess loss of IHO microheater)
      P_pi: 9.6 # mW
      response_time: 970 # ns (rise; fall 980)
```

**候选 2:TFLN_EO_PS(薄膜铌酸锂电光相移器)** — 7mm_0.6uW_1.7ns
- 论文:Electro-optically programmable photonic circuits enabled by wafer-scale integration on thin-film lithium niobate(arXiv, 2023;南开;尚无非期刊版本)
- arXiv: https://arxiv.org/abs/2304.03461(DOI 10.48550/arXiv.2304.03461)
- 参数:length 7000 µm(~7 mm 相移器), width 未报道, insertion_loss 0.15 dB(每 MZI 单元:2×PS+2×耦合器;单移相器传播损耗 ~0.018 dB), P_pi **0.0006 mW**(0.6 µW@1 MHz π 切换;304 fJ/π,由 C=21 fF、Vπ=5.38 V 换算), response_time 1.7 ns(fall;rise 0.5 ns)
- 权衡轴 vs ITO_PS:P_pi 与响应时间数量级更优(0.0006 vs 15 mW、1.7 vs 10000 ns),IL 0.15 略劣、长度 7000 ≫;vs MEMS_PS:IL 与响应更优,P_pi 0.0006 vs 0(注意 MEMS 的 0 也是理想值)→ **对两个真实器件均帕累托有效**
- 对 Customized_PS:全轴被支配
- ⚠️ 预印本未同行评审;P_pi 为换算值(论文给 0.6 µW@1MHz 与 304 fJ/π,换算成 mW 单位)
```yaml
phase_shifter:
  TFLN_EO_PS:
    PDK: ""
    ref: "https://arxiv.org/abs/2304.03461"
    type: PhaseShifter
    exp: True
    cfgs:
      length: 7000 # um (~7 mm phase modulator)
      # width: not reported
      insertion_loss: 0.15 # dB (per MZI-unit: 2 PS + 2 couplers)
      P_pi: 0.0006 # mW (0.6 uW per shifter @ 1 MHz pi-switching; 304 fJ/pi)
      response_time: 1.7 # ns (fall; rise 0.5)
```

**调研备注**:备选 Parra et al. "Ultra-low loss hybrid ITO/Si thermo-optic phase shifter…"(OE 28, 9393, 2020, 10.1364/OE.386959:P_pi 10 mW、50 µm ITO 加热器、响应"几 µs")全文被反爬,未升为完整候选。

### 3.7 无源器件(waveguide / coupler / y_branch / micro_comb)

#### waveguide
**候选 1:WG_SiN_Strip_650nm_VisNIR(全几何报道,保守点)** — 可见-NIR SiN 条波导
- 论文:SiN integrated photonic components in the Visible to Near-Infrared spectral region(OE 32(6), 2024, FBK)
- DOI: https://doi.org/10.1364/OE.514505(arXiv: 2311.16016)
- 参数:mode single, width 650 nm(截面 140×650 nm), type strip, insertion_loss 2.4 dB/cm(TE@750nm;TM 1.6)
- 权衡轴 vs WG_Single_Strip(0.11 dB/cm):IL 高 ~20×,但几何完整报道、不同工艺/波段 → **不支配、真实帕累托点(保守/可见光应用)**
```yaml
waveguide:
  WG_SiN_Strip_650nm_VisNIR:
    PDK: ""
    ref: "https://doi.org/10.1364/OE.514505"
    type: Waveguide
    exp: True
    cfgs:
      mode: single
      width: 650 # nm
      type: strip
      insertion_loss: 2.4 # dB/cm (TE ~750 nm); TM 1.6
```

**候选 2:WG_SiN_FurnaceAnnealed(超低损耗)— ⚠️ 严格支配现有条目**
- 论文:Ultra-Low-Loss Silicon Nitride Photonics Based on Deposited Films Compatible with Foundries(Laser & Photonics Reviews, 2022)
- DOI: https://doi.org/10.1002/lpor.202200544(arXiv: 2301.03053)
- 参数:insertion_loss 0.06 dB/cm(炉退火;退火前 0.42、RTA 后 0.28);width/type 未在摘要报道
- 支配检查:**IL 0.06 < 0.11 严格支配 WG_Single_Strip** → 按帕累托规则应剔除,除非确认 waveguide 条目只比较 IL 轴(该类型无功耗/面积轴,任何新条目都必然支配或被支配——**需你决策 waveguide 是否接受"支配式"扩展**);且 width 缺失 → 仿真器 area 分支抛错,需从论文正文补几何
```yaml
waveguide:
  WG_SiN_FurnaceAnnealed_0p06:
    PDK: ""
    ref: "https://doi.org/10.1002/lpor.202200544"
    type: Waveguide
    exp: True
    cfgs:
      mode: single
      # width: <not reported in abstract; MUST be filled from paper body>
      type: strip
      insertion_loss: 0.06 # dB/cm (furnace-annealed)
```

#### coupler
**候选 1:CPL_MMI_SiN_Angled_PS_80x13(几何完整,实测)** — 宽带 SiN 角度 MMI
- 论文:Low-loss and ultra-broadband silicon nitride angled MMI polarization splitter/combiner(OE, 2020)
- DOI: https://doi.org/10.1364/OE.405188
- 参数:footprint 80×13 µm, insertion_loss ~0.7 dB(TE, C+L 波段;TM ~0.8), crosstalk < −18 dB;**语义注意:这是偏振分束器(PBS)式 2x2 MMI,非标准 3-dB 分束器**
- 权衡轴 vs LT_DC_2x2(5.25×2.4, 0.33 dB):全部轴更差(面积/IL)→ 满足"不支配现有"要求,但作为**可选"更大面积换带宽"**点
```yaml
coupler:
  CPL_MMI_SiN_Angled_PS_80x13:
    PDK: ""
    ref: "https://doi.org/10.1364/OE.405188"
    type: MMI
    exp: True
    cfgs:
      length: 80 # um
      width: 13 # um
      insertion_loss: 0.7 # dB (TE, C+L); TM ~0.8
```

**候选 2:CPL_MMI_SiN_GMMI_2x2(真 3-dB MMI,但缺几何)— ⚠️ 面积缺口**
- 论文:同 Sanna et al. OE 2024(DOI 10.1364/OE.514505)
- 参数:insertion_loss < 0.5 dB@730 nm,输出不平衡 ±0.2 dB;length/width 未在可访问文本报道(论文 Fig.5/Table 1 有)
- 权衡轴 vs LT_DC_2x2:IL 0.5 > 0.33(劣),面积未知 → 不支配;但 **length/width 缺失 → 仿真器抛错**,需从论文补
```yaml
coupler:
  CPL_MMI_SiN_GMMI_2x2:
    PDK: ""
    ref: "https://doi.org/10.1364/OE.514505"
    type: MMI
    exp: True
    cfgs:
      # length: <not reported in accessible text> # um
      # width: <not reported in accessible text> # um
      insertion_loss: 0.5 # dB (upper bound, 730 nm)
```

**备选线索**:Nikbakht et al."Asymmetric, non-uniform 3-dB directional coupler…"(OL 48(2), 2023, DOI 10.1364/OL.476537, arXiv:2208.12148):器件总长 800 µm(含 S-bend),1300-1600 nm,**IL 未在摘要报道**(需全文核实后可用)。

#### y_branch
**候选 1:YJ_Si_Compact_1p2x2(实测,硅工艺)** — 紧凑低损耗 Y 结
- 论文:A compact and low loss Y-junction for submicron silicon waveguide(OE 21(1), 2013)
- DOI: https://doi.org/10.1364/OE.21.001310
- 参数:width 1.2 µm, length 2.0 µm, insertion_loss 0.28±0.02 dB(8 英寸晶圆实测均匀);SOI 亚微米工艺(非 SiN)
- 权衡轴 vs Customized_Y_Branch(IL 0.1):IL 0.28 劣 ~3×,但实测 + 几何完整报道 → 不支配、真实点
```yaml
y_branch:
  YJ_Si_Compact_1p2x2:
    PDK: ""
    ref: "https://doi.org/10.1364/OE.21.001310"
    type: Y_branch
    exp: True
    cfgs:
      width: 1.2 # um
      length: 2.0 # um
      insertion_loss: 0.28 # dB (measured avg)
```

**候选 2:SPL_SiN_GMMI_1x2(1x2 分束,缺几何)— ⚠️ 面积缺口**
- 论文:同 Sanna et al. OE 2024(G-MMI 作 1x2 使用)
- 参数:insertion_loss < 0.5 dB@730 nm;length/width 未报道 → **需补几何才能入库**
```yaml
y_branch:
  SPL_SiN_GMMI_1x2:
    PDK: ""
    ref: "https://doi.org/10.1364/OE.514505"
    type: Y_branch
    exp: True
    cfgs:
      # width/length: <not reported> # um
      insertion_loss: 0.5 # dB (upper bound, 730 nm)
```

#### micro_comb
**候选 1:MC_Si3N4_DKS_100GHz(真实孤子微梳,替换理想条目)** — Nature 2017
- 论文:Microresonator-based solitons for massively parallel coherent optical communications(Nature 546, 274-279, 2017)
- DOI: https://doi.org/10.1038/nature22387(arXiv: 1610.01484)
- 参数:微环半径 240 µm → 直径 ~480 µm(=2R,推导值);谐振器波导截面 0.8×1.65 µm²(耦合点);FSR ~100 GHz(实测 95.75 GHz);Q > 10⁶
- 与 LT_Micro_Comb(理想、无 ref)对比:提供真实出处;模型仅消费 width/length 面积
```yaml
micro_comb:
  MC_Si3N4_DKS_100GHz:
    PDK: ""
    ref: "https://doi.org/10.1038/nature22387"
    type: micro_comb
    exp: True
    cfgs:
      width: 1650 # nm (resonator waveguide width at coupling point, 0.8x1.65 um^2)
      length: 480 # um (resonator diameter = 2 x radius 240 um; derived)
```

**调研备注**:① 无开放获取、带明确 IL 的 SiN Y-branch;硅 YJ(Zhang 2013)是几何完整可用的替代;② 备选真实微梳:B. Stern et al. "Battery-operated integrated frequency comb generator"(Nature 562, 2018, 10.1038/s41586-018-0598-9,开放文本不可取)。

### 3.8 mrr_rerouter(环形开关)
> ⚠️ 同 TIA:仿真器要求 `length/width/area` 必居其一(`utils.py:354-358`),否则抛错。两个候选的论文均未报单环面积 → 入库前需解决(见 3.3 备注)。

**候选 1:MRR_THERMAL(热调,低 IL)** — 严格无阻塞微环开关
- 论文:Ultralow-crosstalk, strictly non-blocking microring-based optical switch(Photonics Research 7(2), 2019)
- DOI: https://doi.org/10.1364/PRJ.7.000155
- 参数:static_power 10 mW/环(半 FSR 调谐 8-10 mW,单通路含 on/off 态约 20 mW,4x4 全载约 80 mW);insertion_loss 0.5 dB/环(on-resonance drop 态,through 态 0.1 dB);消光 >25 dB;串扰抑制 >50 dB;通带 ~24 GHz;热调 rise/fall 14.3/1.2 µs;32 环 fabric 芯片 1.5×2.4 mm²
- 权衡轴 vs LT_MRR_REROUTER:IL 0.5 < 0.93(优),static_power 8-10 ≫ 0.275 mW(劣,~30×)→ 真实帕累托点(IL 轴 vs 功耗轴)
- 支配检查:不被支配(LT 是理想条目,0.275 mW 与 0.93 dB 无法被真实器件同时达到)
```yaml
mrr_rerouter:
  MRR_THERMAL:
    PDK: ""
    ref: "https://doi.org/10.1364/PRJ.7.000155"
    type: ring
    exp: True
    cfgs:
      static_power: 10.0 # per-ring tuning power to half FSR (paper: 8-10 mW); per-path static ~20 mW
      insertion_loss: 0.5 # on-resonance (drop) ring loss, dB; off-resonance through 0.1 dB
      # length/width omitted: per-ring footprint not reported (fabric 1.5x2.4 mm2, 32 rings)
```

**候选 2:MRR_PCM(非易失,零保持功耗)** — PCM 环开关
- 论文:Nonvolatile Electrically Reconfigurable Integrated Photonic Switch Enabled by a Silicon PIN Diode Heater(Adv. Mater. 32(31), 2020)
- DOI: https://doi.org/10.1002/adma.202001218(arXiv: https://arxiv.org/abs/1912.07680)
- 参数:static_power **0 mW** 保持功耗(非易失自保持,论文未给数值,机制性结论);IL 仅定性"near-zero additional loss";面积定性"compact"
- 权衡轴 vs LT_MRR_REROUTER:保持功耗 0 < 0.275(优);IL/面积未量化 → 仅功耗轴帕累托点,**部分表征**
- 支配检查:功耗轴不被支配;IL/面积轴未知(诚实标注)
```yaml
mrr_rerouter:
  MRR_PCM:
    PDK: ""
    ref: "https://doi.org/10.1002/adma.202001218"
    type: ring
    exp: True
    cfgs:
      static_power: 0.0 # nonvolatile self-holding PCM state (no numeric mW in paper)
      # length/width/insertion_loss omitted: not reported numerically
```

**调研备注**:另一候选 8x8 Si3N4 MRR 开关(Photonics Research 2023, 10.1364/PRJ.479499,on 态 IL ~0.37 dB,通带 69 GHz)功耗/面积未在摘要给出;Sherwood-Droz 2008 4x4 hitless 路由器(10.1364/OE.16.015915)全文被墙、数值未核实。**若库要求每条必须有数值化 IL,优先候选 1,候选 2 仅作定性补充。**

## 4. 入库前审阅清单(每条候选必须逐项打勾)

- [ ] `ref` 为可解析的 DOI/arXiv 链接,且指向报道该器件的原始论文
- [ ] 每个 `cfgs` 数值都能在论文中找到出处(摘要/正文);未报道的键已省略并标注
- [ ] 单位与仿真器约定一致(um/nm/dB/mW/GHz/Gbps/fJ/bit/ns/dBm)
- [ ] 满足仿真器消费 schema——**调研新增的硬约束**:
  - [ ] **面积必居其一**:`area` 或 `width×length` 必须存在(`utils.py:354-358`,两者皆缺 → ValueError 崩溃)。TIA/ADC-4P8/NEC_APD/CPL-GMMI/SPL-GMMI/MRR 两候选均卡在此
  - [ ] **类型必填键**:mzm 的 `extinction_ratio`/`efficiency`/`modulation_speed`(`energy.py:161`、`cal_MZM_power`)三者缺一即 KeyError → Si_MZM_TWDD、TFLN_MZM_Hy110 因此不可用;adc 的 `type ∈ {sar, flash}`(其他值 → NotImplementedError)
  - [ ] 缺 `insertion_loss` 的器件按 0 dB 处理(`insertion_loss.py:25-29`)——物理上可能失真,需在注释标注
- [ ] 相对现有变体是帕累托有效(≥1 轴更优、≥1 轴更差);严格支配/被支配 → 剔除(例外:waveguide 等单轴类型需单独决策,见 §3.7;被理想器件支配的情况见 §5.2)
- [ ] 可行域:`out/in_bit ≤ prec`、`work_freq ≤ min(各器件速率上限)` 在当前搜索空间(precision ∈ {2,4,8},work_freq ∈ {5,10,15,20,25})下成立或明确指出解锁效应(ADC_SAR_90GS 解锁 15-25 GHz;ADC_SAR_4P8GS_LP 反而收窄到 4.8)
- [ ] 无物理槽位不合法(如 1x1 光栅进 2x2 槽位、PBS 语义进 coupler 槽位需确认)
- [ ] 变体命名符合 SimPhony 惯例
- [ ] YAML 格式校验通过:`discover_device_catalog` 能加载;analytic 镜像与真实仿真器能消费(`python scripts/experiments/verify_search_space.py --devices mzm,coupler,photodetector --mode analytic` 等)
- [ ] 理想条目/支配问题的处置已决策(见 §5)

## 5. 待决问题(写入库之前需要确认)

1. **落盘方式**:按上游惯例把新变体追加进现有分类文件,还是新增独立 YAML 文件?两者皆可被 glob 加载;追加更贴近上游,独立文件便于标注"本地扩展"。
2. **理想器件处置(核心)— ✅ 已执行(2025-10-21)**:`Customized_PS`(P_pi 0 / resp 0 / 无 ref)、片上 `LT_Laser`(0 IL)、`Customized_Y_Branch`(10 nm 宽)、`Customized_DC_1x1`(槽位不合法)已从**器件搜索空间**移除。实现方式为**发现层过滤**而非删 YAML:`photon_tuner/core/search_space.py` 新增 `EXCLUDED_SEARCH_VARIANTS` 集合,`discover_device_catalog/variants` 不再返回这些条目 → 工具箱、证据探针、LLM 候选均不再暴露;YAML 库与仿真器 readback 保持不动(尊重 vendored 只读约定)。`configs/search_space.json` 快照同步删除。效果:PS 维度在 ITO/MEMS 之间出现真实权衡;on/off-chip 激光开关变为纯效率权衡;y_branch 退为单候选(待补真实条目)。
3. **支配式扩展决策**:waveguide 的 WG_SiN_FurnaceAnnealed(0.06 dB/cm)严格支配现有条目——该类型只有 IL 一个轴,任何新条目都必然支配或被支配;是否接受"支配式更新"(即直接替换最优值),还是坚持"每条都要可权衡"?
4. **搜索空间快照**:`configs/search_space.json` device 层 `candidates` 是文档性快照;入库后是否同步刷新?
5. **验证方式**:入库后跑 `tests/run_tests.py` + `scripts/experiments/verify_search_space.py`(analytic/real 模式)确认加载与探针枚举正常。

## 7. 光子计算核拓扑:现状分析与扩展路径(调研记录)

### 7.1 SimPhony 支持的拓扑机制

层次化 4 级结构,每级网表都是**任意有向无环图(DAG)**:

| 层级 | 配置文件 | 内容 |
|---|---|---|
| 异构架构 | `configs/design/architectures/LT_hetero.yml` | `sub_archs` 列表(可多个,当前仅 LT) |
| 子架构 | `configs/design/sub_architectures/LT.yml` | memory / tiles / cores_per_tile / dataflow / op_type / core |
| 核心网表 | `configs/design/cores/dptc.yml` | DPTC **Crossbar**:实例 i0-i10、`scaling_rules`(R,C,H,W,N 表达式)、`nets`、`replication`(i6 分束树 `i6(s)→i6(e)`、i3 MRR ×2)、`temporal_accum_factor`、`operand_1/2`、`laser_power_distribution`、`chip_mapping`、node 引用 |
| 节点网表 | `configs/design/nodes/dode.yml` | DODE 点积节点:y_branch(i5)→PS(i0)/coupler(i1)→平衡 PD(i2,i3),`nets` n1-n5 |

- 网表消费方式(`onnarchsim/workflow/insertion_loss.py`):`construct_node_graph` / `construct_core_graph` 用 **networkx.DiGraph** 把 instances+nets(+replication 链)建成 DAG → `find_longest_path` 拓扑排序求最长插损路径;`area.py` 按拓扑层级+宽度布局估算面积;能量按实例 `scaling_rules` 表达式计器件数。
- **自由度结论:拓扑本质上是"任意 DAG"**——只要给 `instances + nets(+replication/scaling_rules/ports_num)` 且满足 DAG、端点合法、器件类型被能量公式覆盖,SimPhony 就能计算。
- **但当前库只有一对固定模板**(DODE node + DPTC core):计算范式被锁死在"MZI 交叉阵列 + 点积节点"。

### 7.2 搜索空间中"拓扑"的现状

`configs/search_space.json` 已有 **topology(拓扑与电路层)**,但语义是**电路级参数**,不是图结构:

| 层 | 参数 | 实际改写内容 |
|---|---|---|
| system | tiles / cores_per_tile / dataflow | 平铺与数据流(非光子拓扑) |
| optical_core | core_height / core_width / num_wavelength / work_freq | 交叉阵列维度;网表靠 scaling_rules/replication 随 H/W/N 缩放 |
| **topology** | sharing_policy / temporal_accum / laser_distribution | 仅改写核心网表 3 个字段(ADC/TIA 共享规则、时间累加深度、激光功率分布因子,见 `photon_tuner/core/topology.py`)——**不改变器件连接关系** |
| device | 变体选择 + 实例开关(laser_source / splitter_kind) | 换器件类型/变体,网表结构不变 |

**结论:图结构(器件如何连接、什么计算范式)目前不是搜索维度。** 现有搜索控制的是:维度缩放、电路共享/累加/激光分配、器件变体、平铺/数据流。硬件 dict → sub-arch 的唯一装配点 `build_sub_arch`(`photon_tuner/simulators/analytic.py:187`)固定加载 dptc.yml + dode.yml,analytic 与 real 两条路径都经过它(一处改动两处生效);real 模式还靠 `_inject_hardware`(`photon_tuner/simulators/simphony.py:322`)把重建的 sub-arch 注回 ONNArchSimulator。

### 7.3 扩展拓扑的可行路径(候选)

机制上零阻力(上游原生支持任意 DAG、多 sub-arch,`SubArchitectureLib` 按 `file != default` 加载任意子架构文件):

1. **新增模板文件**:`configs/design/nodes/<node>.yml`、`cores/<core>.yml`、`sub_architectures/<arch>.yml`,并以新 sub-arch 名挂入 `architectures/<hetero>.yml`;
2. **把模板选择变成搜索参数**:search_space 新增一层/参数(如 `node_kind` / `core_kind` / `topology_template`),在 `build_sub_arch` 按该参数选模板——analytic/real 自动同步;
3. **硬约束**:
   - 网表必须是 DAG,`nets` 端点必须都在 `instances` 中;
   - 使用的器件类型必须落在能量公式覆盖内(`dac`/`mzm`/`phase_shifter`/`photodetector`/`on|off_chip_laser`/`adc`/`tia` + 通用 static/dynamic power);新型器件需在 `energy.py` 加公式;
   - `scaling_rules`/`replication` 用 R,C,H,W,N 表达式,与 `temporal_accum`、`chip_mapping` 语义一致;
   - 每模板需有 `insertion_loss_final_device`(节点终点器件)与完整 nets(否则 IL 路径按默认规则退化);
4. **候选拓扑方向**(与器件扩展同一"真实文献驱动"原则,待调研):
   - 节点级:当前 DODE = MZI 点积单元;可加**微环权重节点**(每权重一个 ring modulator,广播-权重范式)、**光电二极管阵列节点**等;
   - 核心级:Crossbar 之外可加 **MZI 网格(Clements/Reck 酉矩阵)**、**广播-权重(broadcast-and-weight)**、蝶形等;
   - 时间累加/空间累加已被 `temporal_accum`/`sharing_policy` 覆盖;
5. **注意**:项目当前路径是"单一 sub-arch 注入"(`layer_to_sub_arch_mapping[layer] = sub_arch`),多 sub-arch 按层类型映射的上游机制(`check_layer_mapping`)存在但未在项目侧接线;若要做"不同层用不同拓扑"需另行接线。

**待决策:是否按 §7.3 做拓扑扩展、选哪个方向(节点级/核心级)、模板选择参数怎么命名与枚举。** 调研与草案先行,确认后再实施(与器件扩展同一流程)。
