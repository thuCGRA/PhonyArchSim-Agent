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


def _isolate_protocol_stdout() -> None:
    """把 JSON-RPC 协议独占到一个私有 fd，并让 fd 1 / ``sys.stdout`` 都指向
    ``/dev/null``。

    必须在**任何重型 import 之前**调用。真实 SimPhony 及其依赖会在 import 阶段
    与运行阶段向 stdout 写普通文本：``[W] Cannot import torchsummary``、
    ``CACTI simulation completed. Parsing the results...``（后者是模拟器里的
    Python ``print``）。MCP 的 stdio 传输把 stdout 当作 JSON-RPC 通道，任何一个
    杂字节都会让对端 ``JSONSerializable`` 解析失败——表现为随机的
    ``Invalid JSON`` / ``JSONDecodeError``，整轮实验被中断。

    关键点：**不能**仅仅把 ``sys.stdout`` 重绑到协议 fd。那样协议和 Python
    ``print`` 又共用同一条流，问题原样复现。正确做法是三个通道彻底分离：

    * 协议   -> 私有 fd（复制自原 fd 1，仅 :func:`_serve` 使用）
    * fd 1   -> ``/dev/null``（吃掉 C 层、子进程、CACTI 的原生输出）
    * ``sys.stdout`` -> ``/dev/null``（吃掉模拟器的 Python ``print``）

    服务端日志仍可经 ``stderr`` 观察，不损失可诊断性。
    """
    global _PROTOCOL_FD
    _PROTOCOL_FD = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    # Python 层输出也丢掉：模拟器的 print 走 sys.stdout，而不是 fd 1。
    sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _serve() -> None:
    """以显式协议流运行 MCP stdio 服务，绕开被隔离的 ``sys.stdout``。"""
    import anyio
    from io import TextIOWrapper

    from mcp.server.stdio import stdio_server

    async def _main() -> None:
        binary = os.fdopen(_PROTOCOL_FD, "wb", buffering=0)
        protocol = anyio.wrap_file(TextIOWrapper(binary, encoding="utf-8"))
        async with stdio_server(stdout=protocol) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )

    anyio.run(_main)


#: 承载 JSON-RPC 的私有 fd（:func:`_isolate_protocol_stdout` 填入）。
_PROTOCOL_FD: int = -1

# 就地进行隔离：位置必须在这里，晚于它 import 的模块可能已经向 fd 1 写过东西。
_isolate_protocol_stdout()

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

    注意：**工作负载由这些环境变量决定**，不是由 ``evaluate`` 的参数决定。
    主机侧的 ``SimphonyAdapter(nn_model_factory=...)`` 在 ``transport="mcp"``
    下不生效——若主机忘了设置 ``SIMPHONY_NN_FACTORY`` / ``SIMPHONY_WORKLOAD``，
    服务端会静默回退到 ``vit_tiny``，于是"mobilevit 跑出来是 vit_tiny 的数"。
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
    """保留为兼容包装；真正的隔离已由 :func:`_isolate_protocol_stdout` 完成。

    进程启动时 fd 1 已指向 ``/dev/null``，协议走私有 fd，因此这里不需要再做
    任何重定向——再做一次反而会把协议 fd 关掉。
    """
    yield


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
    # 隔离已在模块顶部（任何重型 import 之前）完成；这里用显式协议流启动。
    _serve()
