"""SimPhony 评估封装 —— 传统算法基线的统一求值路径。

**与 agent 侧同引擎**：默认 ``mode="real"`` + ``transport="mcp"``，即与
``photon_tuner.cli`` 构造的适配器完全一致——真实 ONNArchSimulator 在 Python-3.10
虚拟环境中运行，本进程（Python-3.14）通过 MCP stdio 桥跨进程调用。基线因此与
agent 走同一条求值路径、同一目标函数、同一份 ``metrics.json``，两者的对比才成立。
（analytic 解析镜像已废弃，只是模拟器库内部供离线实验/单元测试直接调用的能力。）

此外提供：

* **指纹去重** —— 同一硬件的两次求值只跑一次模拟器，同时如实记录缓存命中（文献
  中用于如实地报告「浪费掉的」求值预算）；
* **求值计数** —— ``n_unique``（去重后的不同设计点）与 ``n_calls``（总调用次数）；
* **输出根目录隔离** —— 每个算法/种子的运行有独立的 ``output_root``，避免互相覆盖。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from photon_tuner.common.io import write_json
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.core.schemas import EvaluationResult
from photon_tuner.core.search_space import (
    DesignSpace,
    build_design_space,
    discover_device_catalog,
    materialize_hardware,
)
from photon_tuner.simulators.simphony import SimphonyAdapter

from baselines.action_space import canonical_fingerprint


@dataclass
class EvalRecord:
    """一次求值记录。``cached`` 表示从内部缓存命中（指纹已存在）。"""

    hardware: Dict[str, Any]
    metrics: Dict[str, Any]
    fingerprint: str
    eval_index: int
    cached: bool
    wall_s: float
    run_dir: str
    action: Optional[Dict[str, Any]] = None


class SimphonyBaselineEvaluator:
    """N:1 封装：所有传统算法共用同一个 SimPhony 适配器，共享去重缓存。"""

    def __init__(
        self,
        *,
        simphony_root: Path | str,
        output_root: Path | str,
        mode: str = "real",
        transport: str = "mcp",
        mcp_env: Optional[Dict[str, str]] = None,
        workload_name: Optional[str] = None,
        search_space: Optional[Dict[str, Any]] = None,
        design_space: Optional[DesignSpace] = None,
        model_type: str = "transformer",
        nn_model_factory: Optional[str] = None,
        device_lib_root: Optional[Path | str] = None,
        cache: bool = True,
    ) -> None:
        self.mode = mode
        self.transport = transport
        self.workload_name = workload_name
        self.output_root = Path(output_root).resolve()
        self.search_space = search_space or {}
        self._cache_enabled = cache

        simphony_root = Path(simphony_root).resolve()
        dev_root = Path(device_lib_root or (simphony_root / "configs" / "devices")).resolve()
        self.catalog = discover_device_catalog(str(dev_root))

        self.space = (
            design_space
            or build_design_space(search_space=self.search_space, device_catalog=self.catalog)
        )

        self.adapter = SimphonyAdapter(
            simphony_root=simphony_root,
            output_root=self.output_root,
            search_space=self.search_space,
            mode=mode,
            workload_name=workload_name,
            model_type=model_type,
            nn_model_factory=nn_model_factory,
            transport=transport,
            mcp_env=mcp_env,
        )

        self._cache: Dict[str, EvalRecord] = {}
        self._eval_counter = -1  # 初始求值从 0 开始
        self._total_wall = 0.0

    @property
    def n_unique(self) -> int:
        return len(self._cache)

    @property
    def n_calls(self) -> int:
        return self._eval_counter + 1

    @property
    def total_wall_s(self) -> float:
        return self._total_wall

    def evaluate(
        self, hardware: Dict[str, Any], action: Optional[Dict[str, Any]] = None
    ) -> EvalRecord:
        """求值一个设计点（带归一化、去重与指标扩展）。

        返回的 ``EvalRecord.metrics`` 中已包含 ``objective`` 键（经
        ``metrics_with_objective`` 扩增，含 IL 预算惩罚）。
        """
        hw = self.space.normalize(materialize_hardware(hardware))
        fp = canonical_fingerprint(hw)
        self._eval_counter += 1

        if self._cache_enabled and fp in self._cache:
            # 缓存命中：不重跑模拟器，但如实记录 eval_index 和 cache 标记。
            cached = self._cache[fp]
            return EvalRecord(
                hardware=hw,
                metrics=dict(cached.metrics),
                fingerprint=fp,
                eval_index=cached.eval_index,
                cached=True,
                wall_s=0.0,
                run_dir=cached.run_dir,
                action=action,
            )

        t0 = time.perf_counter()
        result: EvaluationResult = self.adapter.evaluate(hw, self._eval_counter)
        dt = time.perf_counter() - t0
        self._total_wall += dt

        metrics = dict(result.metrics)  # SimphonyAdapter 已在内部调用 metrics_with_objective
        rec = EvalRecord(
            hardware=hw,
            metrics=metrics,
            fingerprint=fp,
            eval_index=self._eval_counter,
            cached=False,
            wall_s=dt,
            run_dir=str(result.run_dir),
            action=action,
        )
        if self._cache_enabled:
            self._cache[fp] = rec
        return rec

    def close(self) -> None:
        self.adapter.close()