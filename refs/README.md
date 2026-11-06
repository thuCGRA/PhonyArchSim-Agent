# 参考文献资料（refs/）

本目录集中存放本项目用到的**原始文献资料**，供器件参数溯源、架构建模依据和文档引用使用。

> ⚠️ 说明：本目录是**本地文献资料**（已在 `.gitignore` 中忽略），**不参与程序运行**，
> 任何代码都不会在运行时读取它。它的作用是给器件库 YAML 与设计文档提供“论文出处”。
> 删除本目录不会影响仿真流程，但会丢失器件参数的出处依据。

## 目录结构

目录为平铺结构，按用途分为两组：

### 一、架构 / 矩阵乘（同行评审全文，PMC-JATS XML）

| 文件 | 论文标题 | 出处 | 用途 |
|---|---|---|---|
| [bai2023.xml](bai2023.xml) | *Microcomb-based integrated photonic processing unit*（Bai 等） | Nature Communications 2022，doi:10.1038/s41467-022-35506-9，PMC9814295 | 微梳光处理单元：光子加速器整体架构建模的参考 |
| [review2022.xml](review2022.xml) | *Photonic matrix multiplication lights up photonic accelerator and beyond*（Zhou 等） | Light: Science & Applications 2022，doi:10.1038/s41377-022-00717-8，PMC8814250 | 光子矩阵乘 / 光加速器综述，建模范式与指标口径参考 |

### 二、相移器 / 可编程光子器件（PDF 原文 + 文本抽取）

| 文件 | 论文标题 | 出处 | 对应项目器件 |
|---|---|---|---|
| [iho2301.00816.pdf](iho2301.00816.pdf) | *Thermo-optic phase shifter based on hydrogen-doped indium oxide microheater*（Tong 等） | Laser & Photonics Reviews 2023，doi:10.1002/lpor.202201032，arXiv:2301.00816 | `IHO_PS`：9.6 mW / 5.5 µm²，10×0.55 µm，IL 0.5 dB，响应 ~970/980 ns |
| [tfln2304.03461.pdf](tfln2304.03461.pdf) | *Electro-optically programmable photonic circuits enabled by wafer-scale integration on thin-film lithium niobate*（Zheng 等） | arXiv:2304.03461（**预印本，未同行评审**） | `TFLN_EO_PS`：7000 µm，IL 0.15 dB，P_π 0.0006 mW，响应 1.7 ns |

#### 文本抽取中间产物

以下 `.txt` 是上面 PDF 转文本的中间结果（不同抽取/清洗版本，部分含多栏乱码，仅供全文检索）：

- `iho_ascii.txt`、`iho_clean.txt`、`iho_fulltext.txt`、`iho_text.txt`、`iho_text_full.txt`
- `tfln_ascii.txt`、`tfln_fulltext.txt`

## 与项目代码的关系

器件库 `configs/devices/photonics/*.yml` 里每个相移器（`IHO_PS`、`TFLN_EO_PS`、`MEMS_PS` 等）
的功耗 / 面积 / 插入损耗 / 响应时间，均取自上述论文。权威的“器件 → 论文 → 数值”对照见：

- [docs/design/device_provenance_audit.md](../docs/design/device_provenance_audit.md)
- [docs/design/round1_device_yaml_review.md](../docs/design/round1_device_yaml_review.md)
- [docs/design/device_library_extension_draft.md](../docs/design/device_library_extension_draft.md)

## 维护提示

- 新增器件或修改器件参数时，请把对应论文放入本目录，并在
  `docs/design/device_provenance_audit.md` 中补上“器件 ↔ 文献 ↔ 数值 ↔ 换算说明”。
- 若某篇文献仅作架构/综述参考（不直接对应单个器件），归入第一组即可。
- 预印本请在表格中注明“预印本”，避免当作已同行评审的实测来源。
- 请勿放入验证码拦截页、访问失败页等无效抓取内容。
