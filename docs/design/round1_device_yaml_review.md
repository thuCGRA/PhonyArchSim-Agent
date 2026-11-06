# 第 1 轮 · 器件扩展第一轮(缩窄版):精确改动清单

> 状态:**✅ 已实施(2025-10-21)**。按用户决策只入库 A 组 5 个(`ADC_SAR_90GS`、`DAC_3`、`DAC_4`、`IHO_PS`、`Sandia_GeWG_PD`);可选 DAC_5/micro_comb 未收。94 测试全绿;快照==运行时;5 变体 analytic 冒烟全部可消费且权衡可见。
> 落库方式:追加到现有每类型单文件;每条新增 `platform:` 标注(呼应"不同器件工艺不同"的顾虑——先让异构可见可审计,一致性/集成成本约束留到后续轮)。
> 铁律:每个 `cfgs` 数值可追溯到论文;未报道键省略并标注;`ref` 为可解析 DOI;仿真器面积硬约束(area 或 width×length 必居其一)。
> **2025-10-21 用户决策**:`TFLN_MZM_HL` 排除(area=0 失真);本轮范围缩窄为"权衡轴 + 拓扑急需依赖"。

## A. 直接入库(5 个 + 2 可选)— 附最终 YAML(含 platform 标注)

### A1. adc — `ADC_SAR_90GS`(器件搜索空间最强权衡轴:解锁 work_freq 15-25 GHz)
- 来源:ISSCC 2014,DOI **10.1109/ISSCC.2014.6757477**;90 GS/s(实测至 100)、667 mW、0.45 mm²、32nm
- 权衡 vs `ADC_SAR_1`(10 GS/s、14.8 mW、2850 µm²):速率 9×(优,解锁 15-25 GHz)→ power ×45 / area ×158(劣)→ 真实帕累托点
```yaml
adc:
  ADC_SAR_90GS:
    PDK: ""
    platform: CMOS-32nm-SOI   # paper: 32nm Digital SOI CMOS
    ref: "https://doi.org/10.1109/ISSCC.2014.6757477"
    type: ADC
    exp: True
    cfgs:
      type: sar
      sample_rate: 90 # GSPS (measured to 100)
      prec: 8 # bit (SNDR 33-36 dB -> ENOB ~5.2-5.7; out_bit 限制见 D2)
      power: 667 # mW
      area: 450000 # um^2 (0.45 mm2)
      tech: 32 # nm
```

### A2. dac — `DAC_3` / `DAC_4`(速率/功耗权衡轴)+ `DAC_5`(可选)
- **DAC_3**:IEEE SSCL 2022,DOI **10.1109/lssc.2022.3205884**;50 GS/s、9b、243 mW、0.2 mm²、FoM 23.2 → vs DAC_1/2:FoM/速率双优,power/area 劣
- **DAC_4**:JSSC 2021,DOI **10.1109/jssc.2021.3057608**;28 GS/s、8b、88 mW、0.03 mm² → vs DAC_2:速率 2×、power 88>50、area 30000>11000
- **DAC_5**(可选):ESSCIRC 2022,DOI **10.1109/esscirc55480.2022.9911426**;56 GS/s、280 mW、0.011 mm² → 纯"速率上限 ×4"轴
```yaml
dac:
  DAC_3:
    PDK: ""
    platform: Intel16   # paper: Intel 16 (~16nm-class FinFET)
    ref: "https://doi.org/10.1109/lssc.2022.3205884"
    type: DAC
    exp: True
    cfgs:
      tech: 16 # Intel 16
      area: 200000 # um^2 (0.2 mm2 active)
      prec: 9 # bit
      power: 243 # mW (base; simulator recomputes via FoM path)
      sample_rate: 50 # GSPS
      FoM: 23.2 # fJ/step
  DAC_4:
    PDK: ""
    platform: CMOS-16nm-FinFET
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
  DAC_5:   # 可选
    PDK: ""
    platform: CMOS-16nm-FinFET
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

### A3. phase_shifter — `IHO_PS`(第 2 轮 MZI 网格的原子单元依赖 + vs ITO/MEMS 真实帕累托)
- 来源:Laser & Photonics Reviews 2023,DOI **10.1002/lpor.202201032**(arXiv:2301.00816);10×0.55 µm、IL 0.5、P_pi 9.6 mW、响应 970/980 ns
- 权衡 vs ITO_PS(50×10、IL 0.03、15 mW、10000 ns)/MEMS_PS(100×45、0.33、0、2000 ns):长度/宽度/P_pi/响应 4 轴优、IL 0.5 劣 → 均帕累托有效
- **拓扑依赖**:MZI 网格(可调 2×2 = 2×coupler + 2×phase_shifter)直接吃这个类型;热光变体正是第 2 轮模板需要的
```yaml
phase_shifter:
  IHO_PS:
    PDK: ""
    platform: strip-waveguide-550nm   # 550-nm strip; 平台以论文为准
    ref: "https://doi.org/10.1002/lpor.202201032"
    type: PhaseShifter
    exp: True
    cfgs:
      length: 10 # um
      width: 0.55 # um (550-nm strip waveguide)
      insertion_loss: 0.5 # dB (excess loss of IHO microheater)
      P_pi: 9.6 # mW
      response_time: 970 # ns (rise; fall 980)
```

### A4. photodetector — `Sandia_GeWG_PD`(所有拓扑的输出探测 + 面积/带宽轴)
- 来源:Opt. Express 19, 24897 (2011),DOI **10.1364/OE.19.024897**;1.3×4 µm(5.2 µm²)、45 GHz、0.8 A/W、暗电流 3 nA
- 权衡 vs `ACADEMIA_Digital_PD`(16×20 µm、27 GHz、1.1 A/W):面积 ≪、带宽 >,responsivity 略低 → 真实帕累托点
```yaml
photodetector:
  Sandia_GeWG_PD:
    PDK: ""
    platform: Ge-on-Si (SOI)
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

### A5. micro_comb — `MC_Si3N4_DKS_100GHz`(可选;第 3 轮 WDM 族依赖 + 基线真实性)
- 来源:Nature 546, 274 (2017),DOI **10.1038/nature22387**(arXiv:1610.01484);环半径 240 µm、FSR ~100 GHz、Q>10⁶
- 价值:替换理想 `LT_Micro_Comb`(无 ref)给基线真实性;第 3 轮 WDM/微梳拓扑直接依赖
```yaml
micro_comb:
  MC_Si3N4_DKS_100GHz:
    PDK: ""
    platform: Si3N4 (SiN)
    ref: "https://doi.org/10.1038/nature22387"
    type: micro_comb
    exp: True
    cfgs:
      width: 1650 # nm (resonator waveguide width at coupling point, 0.8x1.65 um^2)
      length: 480 # um (resonator diameter = 2 x radius 240 um; derived)
```

## B. 搁置(本轮不扩展,附理由)

| 类型 | 草案候选 | 搁置理由 |
|---|---|---|
| tia | TIA_LP / TIA_HB | 面积缺口未解决(仿真器 ValueError);现有 TIA_1 够用;不属第 2 轮拓扑急需 |
| waveguide | WG_SiN_Strip / FurnaceAnnealed | 只有 IL 一个轴 → 必然支配/被支配,无清晰权衡;支配式政策未定;搁置 |
| coupler | CPL_MMI_Angled_PS / GMMI | 现有 LT_DC_2x2 已是 50:50,够第 2 轮 MZI 用;新候选是 PBS 语义或缺几何 |
| y_branch | YJ_Si_Compact / SPL_GMMI | 现有 LT_Y_Branch 够用;新候选仅微增益或缺几何 |
| mrr_rerouter | MRR_THERMAL / MRR_PCM | 面积缺口 + 定性;且第 3 轮拓扑需要的是**新类型** mrr_weight,不是 mrr_rerouter |
| mzm | SOH_MZM(IL≈20 dB 灾难)/ TFLN(area=0)/ Si·Hy110(缺必填键) | 全部不可用或低价值;现有 LT_MZM 够用;第 2 轮 MZI 输入编码用现有件即可 |

## C. 排除记录(明确不入库)

| 变体 | 原因 |
|---|---|
| `TFLN_MZM_HL` | 用户决策:width 缺失 → area=0 失真 |
| `Si_MZM_TWDD` / `TFLN_MZM_Hy110` | 缺 mzm 必读键 → 仿真器 KeyError 崩溃 |

## D. 本轮审阅决策点

1. **A 组 5 个(adc/dac×2/ps/pd)是否确认入库**?`DAC_5`、`micro_comb` 两个可选是否收录?
2. **ADC_SAR_90GS 精度**:ENOB 5.2-5.7,搜索空间 out_bit 上限是否限到 6?
3. **platform 标注字段**是否接受(本轮只标注、不约束;一致性/集成成本约束留第 2-3 轮)?
