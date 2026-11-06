"""光子 DSE 框架的集中式、可移植路径解析。

这里的一切都派生自*本文件*的物理位置，因此项目无论被检出到哪里都能正常工作
（``/path/to/epic``、Windows 挂载路径如 ``\\wsl.localhost\\Ubuntu\\path\\to\\epic``，
或任何其他前缀）。任何内容都不依赖进程 CWD 或硬编码的主机路径。

约定：
- 从下方锚点*惰性*解析路径；绝不将绝对主机路径持久化到结果文件。
- 结果文件改为记录相对于其运行/输出锚点的路径。
"""

from __future__ import annotations

from pathlib import Path

# photon_tuner/common/paths.py
#   parents[0] = common/   parents[1] = photon_tuner/   parents[2] = 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

CONFIG_ROOT = PROJECT_ROOT / "configs"

# 内置的第三方 SimPhony 检出（见 simulators/simphony/VENDORED.md）。
SIMPHONY_ROOT = PACKAGE_ROOT / "simulators" / "simphony"

# 全局、跨运行的历史存储（RAG 记忆）。有意地*不*按运行隔离：
# 它在每次调优运行中持续累积，因此位于共享数据目录。
MEMORY_DB_DIR = PROJECT_ROOT / "memory" / "database"
MEMORY_DB_PATH = MEMORY_DB_DIR / "history.db"
