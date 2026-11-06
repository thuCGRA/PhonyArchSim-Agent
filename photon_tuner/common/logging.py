"""光子 DSE 运行时的统一结构化日志。

提供统一的 ``logging`` 配置，使每个组件都输出时间戳、模块上下文，以及
（出错时）完整堆栈。可通过 ``PHOTON_TUNER_LOG_LEVEL`` 覆盖默认的 ``INFO``
级别，例如 ``PHOTON_TUNER_LOG_LEVEL=DEBUG`` 可对 agent/工具执行进行详细跟踪。
"""

from __future__ import annotations

import logging
import os
import sys

_LEVEL = os.environ.get("PHOTON_TUNER_LOG_LEVEL", "INFO").upper()
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger("photon_tuner")
    root.setLevel(getattr(logging, _LEVEL, logging.INFO))
    root.addHandler(handler)
    # 不向进程根 logger 传播，否则会重复输出并引入第三方 handler。
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """返回命名空间位于 ``photon_tuner`` 之下的 logger。"""
    _configure()
    return logging.getLogger(f"photon_tuner.{name}")


def configure_log_level(level: str) -> None:
    """在运行时设置 ``photon_tuner`` logger 级别（例如 ``DEBUG`` 对应 ``--verbose``）。

    幂等，且在 ``get_logger`` 已配置根 handler 之后调用也安全。
    """
    _configure()
    logging.getLogger("photon_tuner").setLevel(
        getattr(logging, str(level).upper(), logging.INFO)
    )
