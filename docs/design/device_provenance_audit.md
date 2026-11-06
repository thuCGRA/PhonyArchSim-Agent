# 器件库逐器件溯源审计(Device Provenance Audit)

这是 criterion (5)「每个 cfgs 数字可追溯至真实论文」的逐器件落地记录。核心原则:
**只填真实可解析 DOI;找不到的字段明确省略/标注,绝不猜 DOI、绝不凭空捏造。**
对应机器检查见 `photon_tuner/core/provenance.py`,测试见
`tests/test_provenance.py::test_every_baseline_device_has_explicit_note`。

## 总览

| 桶 | 数量 | 含义 |
|---|---|---|
| **traceable** | 27 | YAML 带 `ref`(DOI 全部 Crossref 可解析) |
| **baseline** | 27 | 可搜索但 `ref` 为空 —— 逐器件诚实标注(见下表) |
| **placeholder** | 4 | `EXCLUDED_SEARCH_VARIANTS`(理想化结构占位,非物理器件) |
| **合计** | **58** | 第 1 轮 41 + 第 2 轮 13 + 第 3 轮构造 4 |

## 第 3 轮:前沿"轴"订正 + 4 个物理构造变体

本轮不改任何论文实测值(除了下述一处非物理订正),目标是让**已经存在**的实测数字
真正进入帕累托权衡,并只在包络确实有空白处才构造。

### 3.1 前沿轴订正(零构造,`photon_tuner/core/device_frontier.py`)

`_energy()` 原先只读 `power`/`static_power` 且不区分器件类型,导致三类器件无法体现权衡:

| 类型 | 原读取 | 问题 | 订正为 |
|---|---|---|---|
| `tia` | `power`/`static_power` → **None** | 只记录 `dynamic_power` → 点落不到平面上,前沿**为空** | 回退链补 `dynamic_power` |
| `mzm` | `static_power: 0` → 0 | 全类同在 power=0 线上 → 支配退化成"只比面积",前沿塌成单点 | `efficiency`(fJ/bit) |
| `phase_shifter` | `static_power: 0` → 0 | 同上 | `P_pi`(mW) |

输出键由 `power_mW` 改为**单位中立**的 `energy_cost` + `energy_unit` + `energy_field`。
因为前沿是**逐器件类型独立计算**的,同一轴上所有点必然是同一度量,**跨类型绝不混比**。
订正后实测(tradeoffs 全部来自已有论文值,无一个新数字):

| 类型 | 订正前前沿 | 订正后前沿 |
|---|---|---|
| tia | 空 | `TIA_1`(3mW/50µm²) ↔ `TIA_LP`(1.95mW/2000µm²) |
| photodetector | `ACADEMIA_Digital_PD` 单点 | `ACADEMIA_Digital_PD` ↔ `LT_PD`(1.1mW/40µm²) |
| mzm | `LT_MZM` 单点 | `LT_MZM`(450fJ/bit,5200µm²) ↔ `SL_MZM`(50fJ/bit,6250µm²) |
| phase_shifter | `IHO_PS` 单点 | `IHO_PS`(9.6mW/5.5µm²) ↔ `MEMS_PS`(10nW,4500µm²) |

注:`TFLN_EO_PS` 在该二维平面上仍被 `MEMS_PS` 支配——它真正的优势是
`response_time`(1.7ns vs 2000ns),而速度不在本前沿坐标轴内。

### 3.2 非物理值订正(1 处)

`phase_shifter/MEMS_PS` 的 `P_pi` 上游记 `0`(注释写 "nW level"),字面为零在最小化
功耗轴上不物理,已订正为 `0.00001 mW`(10 nW 级)。该器件是**默认 node 相移器**,故
默认配置的能耗与目标值随之微增,两个 golden 测试已同步更新并注明原因:
`tests/test_workload.py`(1699934.16 → 1699935.63 pJ)、
`tests/deep/test_objective_alignment.py`(73137.0809 → 73137.1459)。

### 3.3 物理构造变体(4 个,`baseline` 桶)

dac/adc 的 power×area 包络确有空白,故按"锚定库内真实器件做同量级外推"构造 4 个变体。
它们 **`ref: ""`**(归入 `baseline`,绝不冒充论文实测值),YAML 内以 `# 构造` 注释 +
`note:` 标明,并在 `provenance.py::BASELINE_PROVENANCE_NOTES` 逐条登记(测试强制全覆盖)。

| 变体 | 物理锚点 | 构造 cfgs | 实测效果 |
|---|---|---|---|
| `dac/DAC_LP_28nm` | `DAC_4`(16nm,28GS/s,88mW,0.03mm²) | 20GS/s, 8b, 45mW, 90000µm², 28nm | 20GHz 前沿 2→4 点 |
| `dac/DAC_HS_16nm` | `DAC_5`(16nm,56GS/s,280mW,0.011mm²) | 60GS/s, 8b, 300mW, 8000µm², 16nm | **解锁 50-60GHz**(此前该区间无可用 DAC) |
| `adc/ADC_SAR_2GS_LP` | `ADC_SAR_1`(14nm,10GS/s,14.8mW,0.00285mm²) | 2GS/s, 8b, 1.5mW, 9000µm², 28nm | 2GHz 前沿 2→4 点 |
| `adc/ADC_SAR_30GS` | `ADC_SAR_1` / `ADC_SAR_90GS` 之间 | 30GS/s, 8b, 60mW, 2500µm², 16nm | 20-30GHz 前沿点由 667mW/0.45mm² 变为 60mW/0.0025mm² |

结论:13+4=17 个新增中,真正改善 power×area 前沿结构的是 `DAC_5`、`DAC_HS_16nm`、
`DAC_LP_28nm`、`ADC_SAR_4P8GS_LP`、`ADC_SAR_2GS_LP`、`ADC_SAR_30GS`;而 tia/
photodetector/mzm/phase_shifter 的折中**本来就存在**,只是此前被错误的轴定义隐藏。

## 第 2 轮器件扩展(13 个,2025 新增)

第 2 轮新增 13 个变体,**全部带可解析 `ref`**(均归入 `traceable`)。其中 5 个的
`cfgs` 完全来自论文;8 个论文未报道几何/面积等**非核心物理**键,按"折中构造"补齐,
并在 YAML 内以 `# 折中构造…` 注释 + `note:` 字段逐项标明。**不构造核心物理键**:缺
`extinction_ratio`/`efficiency`/`modulation_speed` 的 MZM 候选直接排除,而非臆造。

### 2.1 A 组:论文参数完整(5 个)

| 变体 | 出处(DOI) | 关键 cfgs(全部论文值) |
|---|---|---|
| `dac/DAC_5` | ESSCIRC 2022 `10.1109/esscirc55480.2022.9911426` | 56 GS/s、8 bit、280 mW、area 11000 µm²、16 nm |
| `micro_comb/MC_Si3N4_DKS_100GHz` | Nature 2017 `10.1038/nature22387` | 半径 240 µm(直径 480 µm,推导)、耦合点 0.8×1.65 µm² |
| `waveguide/WG_SiN_Strip_650nm_VisNIR` | Opt. Express 2024 `10.1364/OE.514505` | strip、650 nm、2.4 dB/cm(TE@750 nm) |
| `y_branch/YJ_Si_Compact_1p2x2` | Opt. Express 2013 `10.1364/OE.21.001310` | 1.2×2.0 µm、IL 0.28 dB(实测均值) |
| `coupler/CPL_MMI_SiN_Angled_PS_80x13` | Opt. Express 2020 `10.1364/OE.405188` | 80×13 µm、IL 0.7 dB(TE);**PBS 语义**,非标准 3-dB 耦合器 |

### 2.2 B 组:论文有 DOI、仅缺几何/面积 → 折中构造并标注(8 个)

| 变体 | 出处(DOI) | 论文值 | 折中构造值(YAML `note:` 已标明) |
|---|---|---|---|
| `mzm/TFLN_MZM_HL` | Nat. Photonics 2019 `10.1038/s41566-019-0378-6` | 3000 µm、IL 2.5 dB、ER 40 dB、eo-BW 70 GHz、112 Gbps | width 460 µm(占位宽,对齐 ACADEMIA MZM 量级);efficiency 700 fJ/bit 为**论文作者估算** |
| `mzm/SOH_MZM` | Sci. Rep. 2018 `10.1038/s41598-017-19061-8` | 1100 µm、ER 14 dB、25 GHz、100 Gbps、98 fJ/bit | width 50 µm;IL 20 dB 为**论文损耗预算分项加总** |
| `phase_shifter/TFLN_EO_PS` | arXiv:2304.03461(**预印本,未同行评审**) | 7000 µm、IL 0.15 dB、P_π 0.0006 mW、1.7 ns | width 1.0 µm(TFLN 条形波导典型宽) |
| `mrr_rerouter/MRR_THERMAL` | Photonics Res. 2019 `10.1364/PRJ.7.000155` | 10 mW/环、IL 0.5 dB | 335×335 µm =√(fabric 1.5×2.4 mm² ÷ 32 环)**推导** |
| `adc/ADC_SAR_4P8GS_LP` | JSSC 2026 `10.1109/jssc.2026.3677867` | 4.8 GS/s、8 bit、10.4 mW、28 nm | area 6000 µm²(**必须补**,否则 area 解析抛错) |
| `tia/TIA_LP` | MWSCAS 2016 `10.1109/MWSCAS.2016.7870132` | 1.95 mW、7 GHz、130 nm(**版图后仿真**) | area 2000 µm² |
| `tia/TIA_HB` | JSSC 2012 `10.1109/JSSC.2011.2178723` | 9 mW、30 GHz、45 nm SOI | area 5000 µm² |
| `photodetector/NEC_APD` | Electron. Lett. 2006 `10.1049/el:20061700` | BW 20 GHz、sensitivity −30.2 dBm、QE 94% | length 30 µm、width 10 µm、responsivity 1.17 A/W(由 QE 换算) |

### 2.3 第 2 轮明确排除(6 个,未入库)

| 变体 | 排除理由 |
|---|---|
| `mzm/Si_MZM_TWDD`、`mzm/TFLN_MZM_Hy110` | 缺 MZM 核心物理键(ER/efficiency/modulation_speed),构造即造假核心物理 |
| `waveguide/WG_SiN_FurnaceAnnealed` | IL 0.06 严格支配现有 0.11(单轴类型,违反帕累托政策) |
| `mrr_rerouter/MRR_PCM` | 仅定性描述(非易失/零保持),无数值 |
| `coupler/CPL_MMI_SiN_GMMI_2x2`、`y_branch/SPL_SiN_GMMI_1x2` | 仅 IL 上界,几何全无 → 无任何构造依据 |

## 27 个 baseline 器件的逐器件标注

外部检索结果:对 DAC_1、SL_MZM、ACADEMIA_* MZM、TO 相移器、微梳分别做 Crossref
关键词检索,均**未找到参数吻合的单器件论文**(抽象不列全 cfgs;命中项多为无关
或不同规格)。因此**不给任何 baseline 器件补编 DOI**,改为逐器件明确标注来源归属,
`note` 已写入 `provenance.py::BASELINE_PROVENANCE_NOTES` 并由测试强制全覆盖。

### A. SimPhony "LT" 基线设计(框架内置基线,无单器件论文)

这组是 SimPhony "LT" 子架构(Hamerly-2019 启发 crossbar)的内置器件,上游未记录
单器件 DOI;批量出处为 SimPhony(arXiv:2411.13715)。

| 器件 | 关键 cfgs | 归属 |
|---|---|---|
| `coupler/LT_DC_2x2` | 5.25 µm, IL 0.33 dB | LT 基线定向耦合器 |
| `mzm/LT_MZM` | 260 µm, IL 1.2 dB, ER 10 dB, 450 fJ/bit @25G | LT 基线 MZM |
| `photodetector/LT_PD` | 1.1 mW dynamic | LT 基线探测器 |
| `micro_comb/LT_Micro_Comb` | 1184 nm × 1184 µm | LT 基线微梳 |
| `mrr_rerouter/LT_MRR_REROUTER` | IL 0.93 dB, 0.275 mW | LT 基线微环 |
| `y_branch/LT_Y_Branch` | IL 0.1 dB | LT 基线 Y 分支 |
| `off_chip_laser/LT_Laser` | WPE 0.3, +2 dB 耦合 | LT 片外激光模型 |

### B. 框架结构占位(非物理器件,按设计无论文)

| 器件 | 关键 cfgs | 归属 |
|---|---|---|
| `coupler/Customized_DC_2x2` | 31 µm, IL 0.33 dB | 结构默认定向耦合器 |
| `laser_splitter/Customized_Laser_Splitter` | 1.3 µm, IL 0.1 dB | 结构默认分束器 |
| `mmi/Customized_1x10_MMI` | 34.6 µm, IL 0.199 dB | 结构默认 1×10 MMI |
| `on_chip_laser/Customized_Laser` | WPE 0.2 | 片内激光模型 |
| `large_insertion/Large_Insertion_Test` | 50 dB IL, −45 dB 串扰 | 测试夹具(非物理) |

### C. "ACADEMIA" 学术参考模型(上游留空 `ref`,论文未定位)

| 器件 | 关键 cfgs | 归属 |
|---|---|---|
| `coupler/ACADEMIA_DC_2x2` | 200 µm, IL 0.5 dB | 学术 DC 模型 |
| `crossing/ACADEMIA_WG_Crossing` | 100 µm, IL 0.25 dB, −45 dB | 学术波导交叉 |
| `crossing/ACADEMIA_WG_Crossing_Linear` | 21 µm, IL 0.13 dB, −35 dB | 学术波导交叉 |
| `crossing/ACADEMIA_WG_Crossing_Trigonometric` | 8 µm, IL 0.23 dB, −30 dB | 学术波导交叉 |
| `mzm/ACADEMIA_EO_MZM_1x1` | 1600 µm, 450 fJ/bit @50G, ER 15 dB | 学术 MZM 模型 |
| `mzm/ACADEMIA_EO_MZM_2x2` | 1600 µm, 450 fJ/bit @25G, ER 15 dB | 学术 MZM 模型 |
| `phase_shifter/ACADEMIA_TO_PS` | P_π 30 mW, 10 µs | 学术热光相移器 |

### D. 电子器件 / 通用 / 未解析

| 器件 | 关键 cfgs | 归属 |
|---|---|---|
| `dac/DAC_1` | 16 nm, 12 bit, 14 GS/s, 169 mW, FoM 33.6 | 上游未记录 DOI;Crossref 未定位到同规格(16nm/12bit/14GS/s) |
| `integrator/Integrator_1` | 560 µm², 0.3 mW | 上游未记录 DOI |
| `waveguide/WG_Single_Strip` | 500 nm, 0.11 dB/cm | 标准 SOI 条形波导(通用技术,无单一论文) |
| `mzm/SL_MZM` | 250 µm, 50 fJ/bit @25G, ER 6 dB | "SL" 前缀未解析;单器件 DOI 未定位 |
| `dac/DAC_LP_28nm` | 28 nm, 8 bit, 20 GS/s, 45 mW, 0.09 mm² | **第 3 轮物理构造**:锚 `DAC_4` 外推(见 §3.3) |
| `dac/DAC_HS_16nm` | 16 nm, 8 bit, 60 GS/s, 300 mW, 0.008 mm² | **第 3 轮物理构造**:锚 `DAC_5` 外推(见 §3.3) |
| `adc/ADC_SAR_2GS_LP` | 28 nm, 8 bit, 2 GS/s, 1.5 mW, 0.009 mm² | **第 3 轮物理构造**:锚 `ADC_SAR_1` 外推(见 §3.3) |
| `adc/ADC_SAR_30GS` | 16 nm, 8 bit, 30 GS/s, 60 mW, 0.0025 mm² | **第 3 轮物理构造**:锚 `ADC_SAR_1`/`90GS` 之间(见 §3.3) |

## traceable 27 个(DOI 全部 Crossref 可解析)

见 `photon_tuner/simulators/simphony/configs/devices/*/*.yml` 中带 `ref` 的条目。
已核对(示例):Reck `10.1103/PhysRevLett.73.58`、Clements `10.1364/OPTICA.3.001460`、
IHO_PS `10.1002/lpor.202201032`、MEMS_PS/ITO_PS `10.1038/s41378-023-00498-z`、
ADC_SAR_1 `10.1109/isscc42614.2022.9731625`、Sandia_GeWG_PD `10.1364/OE.19.024897`
等。第 2 轮新增 13 个的 DOI 见 §2.1/§2.2,均已逐个经 Crossref 返回 200 且标题吻合。

## 违规修正记录

- **Butterfly 网格引用撤回**(Round 12):Tang 2021 `10.1021/acsphotonics.1c00419` 经
  Crossref 复核为「Ten-Port Unitary Optical Processor」(10 端口**万能**处理器),**非**
  非万能 butterfly/FFT 网格;已撤回并把 `ref` 置空标注 "citation under review"。
- **ADC_SAR_1 引用订正**(R1,已批准):补 ISSCC 2022 DOI。
- **MEMS_PS `P_pi` 非物理值订正**(第 3 轮):上游记 `0`(注释 "nW level"),字面为零在
  最小化功耗轴上不物理(且会让该器件在支配判定中显得"零代价")。订正为 `0.00001 mW`
  (10 nW 级);因其为默认 node 相移器,两个 golden 目标值已同步更新(见 §3.2)。
- **前沿"轴"订正**(第 3 轮,零构造):`_energy()` 原先不区分器件类型、只读
  `power`/`static_power`,使 tia 前沿为空、mzm/phase_shifter 前沿塌成单点。改为按类型
  取物理正确的能量度量(见 §3.1),输出键改为单位中立的 `energy_cost`/`energy_unit`。
  未新增或改动任何论文实测值。
