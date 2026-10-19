# Vendored: SimPhony（第三方，请勿随意修改）

本目录是上游 [ScopeX-ASU/SimPhony](https://github.com/ScopeX-ASU/SimPhony) 项目的本地拷贝，
供 real 模式（`onnarchsim.simulator.ONNArchSimulator`）仿真使用。

- 上游来源：https://github.com/ScopeX-ASU/SimPhony
- 论文：arXiv:2411.13715（ACM/IEEE DAC 2025）
- 协议：见本目录 `LICENSE`
- 维护方式：只读引用。如需升级，请整体替换本目录，并在此记录版本 / commit。

## 迁移注意事项

- `cacti/cacti` 是**编译好的 Linux x86_64 可执行文件**，换 OS / 芯片架构后需要在目标机重新编译
  （源码就在本目录 `cacti/` 下，执行 `make` 即可）。
- `cacti/cacti`、`cacti/cacti.i`、`cacti/obj_dbg/`、`runs/` 均为编译 / 运行产物，
  已在项目根 `.gitignore` 中忽略，不入版本库。
