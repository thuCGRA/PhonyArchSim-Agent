"""围绕 :meth:`DeepAgentTuner.tune` 的可选 FastAPI HTTP 封装。

FastAPI / uvicorn 不是硬性运行时依赖：框架可完全离线地从 CLI 运行。
本模块仅按需导入它们，因此 HTTP 服务是可选部署模式，而非必需项。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TuneRequest(BaseModel):
    """``POST /tune`` 的请求体 —— 调用方每次请求可设置的参数。"""

    initial_hardware: Dict[str, Any] = Field(default_factory=dict)
    iterations: int = Field(default=4, ge=1)
    state_strategy: str = "best"
    search_strategy: str = "joint"
    il_budget: Optional[float] = None
    topology_sweep: bool = False
    trajectory_logging: bool = True


def create_app(tuner: Any, adapter: Any = None, store: Any = None):
    """构建一个 FastAPI 应用，通过 ``tuner`` 暴露 ``POST /tune`` 接口。

    ``tuner`` 是一个 ``DeepAgentTuner``（或任何具有兼容 ``tune(**kwargs)``
    方法的对象）。``adapter`` / ``store`` 是保存在应用上的可选句柄，供调用方在
    关闭时释放；它们不会在每次请求后关闭，因为 HTTP 服务器会在多个请求间复用它们。
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - 可选依赖
        raise ImportError(
            "fastapi is required to serve photonic-tuner; install it with "
            "`pip install fastapi uvicorn`"
        ) from exc

    app = FastAPI(title="photonic-tuner", version="0.1.0")
    app.state.tuner = tuner
    app.state.adapter = adapter
    app.state.store = store

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/tune")
    def tune(req: TuneRequest) -> Dict[str, Any]:
        try:
            summary = tuner.tune(
                initial_hardware=req.initial_hardware or None,
                max_iterations=req.iterations,
                state_strategy=req.state_strategy,
                trajectory_logging=req.trajectory_logging,
                il_budget=req.il_budget,
                search_strategy=req.search_strategy,
                topology_sweep=req.topology_sweep,
            )
        except Exception as exc:  # noqa: BLE001 — 以 HTTP 500 状态码返回
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return {
            "iterations": summary["iterations"],
            "best": summary["best"],
            "best_hardware": summary["best_hardware"],
            "run_dir": summary["run_dir"],
            "rework_total": summary.get("rework_total"),
        }

    return app
