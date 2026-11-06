"""SimPhony MCP 服务器 —— 运行在 Python-3.10 虚拟环境下。

暴露一个由真实 ``ONNArchSimulator``（torch 1.12.1 + torchonn + CACTI）支撑的
``evaluate(hardware)`` 工具。配置通过环境变量传入，并带项目本地默认值：

    SIMPHONY_WORKLOAD        工作负载名称（默认 ``vit_tiny``）
    SIMPHONY_NN_FACTORY      模型构建器的 ``module:function`` 写法
                             （默认 ``tests.factory:build_vit_tiny``）
    SIMPHONY_OUTPUT_ROOT     真实模式报告的落盘目录
    SIMPHONY_ARCH_CFG_FILE   子架构 YAML（相对代码库根目录）
    SIMPHONY_MODEL_TYPE      ``transformer`` | ``cnn``

直接运行可做一次手动冒烟测试::

    .venv-simulator/bin/python -m photon_tuner.bridge.mcp.simphony_server
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

# 无论子进程从何处启动，都保证项目根目录与内置 SimPhony 代码库可被导入。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SIMPHONY_ROOT = _PROJECT_ROOT / "photon_tuner" / "simulators" / "simphony"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SIMPHONY_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".tools" / "mplconfig"))

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("simphony")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else value


_adapter = None


def _get_adapter():
    """惰性（且仅一次）构建真实模式适配器。

    ONN 模型在每次 ``evaluate`` 调用时通过 ``_resolve_nn_model`` 重新构建，
    与已验证的进程内真实模式路径保持一致。
    """
    global _adapter
    if _adapter is None:
        from photon_tuner.common.io import read_json
        from photon_tuner.simulators.simphony import SimphonyAdapter

        search_space = read_json(_PROJECT_ROOT / "configs" / "search_space.json")
        _adapter = SimphonyAdapter(
            simphony_root=str(_SIMPHONY_ROOT),
            output_root=_env("SIMPHONY_OUTPUT_ROOT", str(_PROJECT_ROOT / ".runs_mcp" / "real")),
            search_space=search_space,
            mode="real",
            nn_model_factory=_env("SIMPHONY_NN_FACTORY", "tests.factory:build_vit_tiny"),
            model_type=_env("SIMPHONY_MODEL_TYPE", "transformer"),
            arch_cfg_file=_env(
                "SIMPHONY_ARCH_CFG_FILE", "configs/design/architectures/LT_hetero.yml"
            ),
            workload_name=_env("SIMPHONY_WORKLOAD", "vit_tiny"),
        )
    return _adapter


@contextlib.contextmanager
def _mute_stdout():
    """在块内将 fd 1（stdout）指向 fd 2（stderr）。

    真实 SimPhony 模拟器和 CACTI 子进程都会向 stdout 打印进度，若不处理会
    破坏 MCP 的 stdio JSON-RPC 流（客户端把每一行 stdout 都当作协议消息解析）。
    在文件描述符层面重定向，可同时覆盖 Python ``print`` 与子进程输出。
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        sys.stdout.flush()


@mcp.tool()
def evaluate(hardware: str, iteration: int = 0) -> str:
    """用真实 ONNArchSimulator 评估一个光子硬件配置点。

    Args:
        hardware: 完整硬件字典的 JSON 字符串（架构参数 + 器件选择 + 每个器件的
            ``cfgs`` 覆盖项）。
        iteration: 运行序号，仅用于命名服务器端报告目录。

    Returns:
        JSON 字符串 ``{"metrics": {...}, "hardware": {...}, "report": {...}}``。
        ``report`` 是适配器写入其 ``output_root`` 的完整 SimPhony 报告，以带内
        方式返回，使宿主端能落盘生成其读取工具所依赖的相同明细文件。
    """
    hw = json.loads(hardware)
    with _mute_stdout():
        result = _get_adapter().evaluate(hw, int(iteration))
    report = {}
    report_path = _get_adapter().output_root / f"iter_{int(iteration):03d}" / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    return json.dumps({"metrics": result.metrics, "hardware": result.hardware, "report": report})


@mcp.tool()
def ping() -> str:
    """存活探测。"""
    return "pong"


if __name__ == "__main__":
    mcp.run(transport="stdio")
