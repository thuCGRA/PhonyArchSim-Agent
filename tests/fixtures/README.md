# 测试基线数据（tests/fixtures/）

本目录存放**必须随仓库提交**的测试输入。它们原先直接引用 `runs/run_agent/` 下的
运行产物，而 `.gitignore` 里的 `runs/*` 规则把整个 `runs/` 排除了 —— 于是 CI
全新检出（`actions/checkout`）后这些文件并不存在，依赖它们的测试在 GitHub 上
稳定失败，本地却因为留有历史运行记录而通过。

## recorded_runs/

真实运行记录的 `iter_000` 起点（只保留 JSON 证据，丢弃 `simphony.log` 等日志）：

| 目录 | 范式 | 被谁使用 |
|---|---|---|
| `run_iteration_007/iter_000` | `dptc_crossbar` | `tests/test_mesh_eval.py`（主导器件 = DAC） |
| `run_iteration_012/iter_000` | `mzi_mesh_clements` | `tests/test_mesh_eval.py`（主导器件 = laser） |
| `run_iteration_013/iter_000` | `mzi_mesh_clements` | `tests/deep/test_evaluate.py`（损耗/功率预算证据包） |

## 约定

- **不要**从 `runs/` 里隐式读取测试输入；测试所需的一切都在本目录下显式提交。
- 新增/更新基线时，请同步复制必要的 JSON，并在上表登记用途。
- 这里的文件是**输入**，不是输出；运行测试不会写回本目录。
